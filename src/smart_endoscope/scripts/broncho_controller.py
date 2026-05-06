"""
broncho_controller.py
=====================
Chapter 11 – Robot Control integration for the Smart Bronchoscope.

Two main classes are exposed:

    CollisionGuard   – Manual-mode safety layer.
                       Uses the Space Jacobian to project every commanded
                       joint-velocity vector onto the safe sub-space
                       (wrenches that move the tip *away* from walls).
                       Implements the task-space PI feedback law from §11.3.3
                       to damp motion toward detected obstacles.

    AutoPilot        – Autonomous bronchoscope navigation.
                       Implements a simplified computed-torque / task-space
                       feedforward+PI controller (§11.4.3 / §11.3.3).
                       The scope advances through the airway, continuously
                       sampling a virtual "proximity field" from the vision
                       pipeline and reacting with corrective bends so it
                       never touches a wall.

Both classes are stateless between calls (they carry their own integrators)
and are designed to be dropped into the SmartDashboardGUI update loop with
minimal coupling.
"""

import numpy as np
from broncho_kinematics import BronchoKinematics


# ---------------------------------------------------------------------------
# Tunable parameters (edit here, not inside the classes)
# ---------------------------------------------------------------------------

# CollisionGuard
GUARD_WALL_THRESHOLD   = 0.08   # metres – repulsion activates inside this radius
GUARD_KP               = 1.20   # proportional gain for wall-avoidance twist (§11.3.1)
GUARD_KI               = 0.35   # integral gain
GUARD_DAMP             = 0.80   # velocity damping when near wall (0–1 scale)
GUARD_MAX_CORRECTION   = 0.25   # max corrective angle added per call (rad)

# AutoPilot
AUTO_INSERT_SPEED      = 0.008  # m per tick – nominal forward insertion rate
AUTO_BEND_KP           = 1.80   # proportional gain for lateral error (task-space PI)
AUTO_BEND_KI           = 0.40   # integral gain
AUTO_MAX_BEND          = 1.45   # hard joint limit for auto bending (rad)
AUTO_GOAL_THRESHOLD    = 0.02   # metres – stop when tip is this close to the goal
AUTO_WALL_ZONE         = 0.12   # metres – wall repulsion zone in AUTO mode
AUTO_REPULSE_GAIN      = 2.50   # how aggressively to bend away from walls


# ---------------------------------------------------------------------------
# Helper: thin wrapper so CollisionGuard / AutoPilot don't need the full GUI
# ---------------------------------------------------------------------------

class RobotState:
    """Plain data-class carrying the four joint values."""
    def __init__(self, insertion=0.0, proximal=0.0, mid=0.0, distal=0.0):
        self.insertion = insertion
        self.proximal  = proximal
        self.mid       = mid
        self.distal    = distal

    @property
    def as_list(self):
        return [self.insertion, self.proximal, self.mid, self.distal]

    def clipped(self):
        """Return a new RobotState with physically valid joint ranges."""
        return RobotState(
            insertion = np.clip(self.insertion, -0.50,  1.50),
            proximal  = np.clip(self.proximal,  -0.60,  0.60),
            mid       = np.clip(self.mid,       -1.00,  1.00),
            distal    = np.clip(self.distal,    -1.57,  1.57),
        )


# ---------------------------------------------------------------------------
# Proximity reading (vision-based obstacle sensing)
# ---------------------------------------------------------------------------

def estimate_wall_proximity(frame, tip_pos_2d):
    """
    Lightweight obstacle sensor that works on the live camera frame.

    Returns
    -------
    dist   : float  – estimated distance to nearest wall (metres, clamped >0)
    normal : ndarray[2]  – unit vector pointing away from the nearest wall,
                           expressed in the camera / task-space XY plane.
    margin : float  – lateral clearance score [0=touching, 1=fully centred]

    The method uses simple image-moment analysis on the dark-region mask
    (the airway wall tends to be darker at the periphery).  This is
    intentionally kept fast and dependency-free so it runs every GUI tick.
    """
    import cv2

    if frame is None:
        return 0.5, np.array([0.0, 1.0]), 1.0

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Convert to grayscale and threshold for dark regions (walls)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Crop to a central ROI so we only sense what's ahead of the tip
    roi_size = min(h, w) // 3
    x0 = max(0, cx - roi_size)
    x1 = min(w, cx + roi_size)
    y0 = max(0, cy - roi_size)
    y1 = min(h, cy + roi_size)
    roi = dark_mask[y0:y1, x0:x1]

    dark_ratio = roi.mean() / 255.0  # 0 = all bright (open), 1 = all dark (wall)

    # Estimate direction of nearest wall from image moments of the dark region
    moments = cv2.moments(dark_mask)
    if moments["m00"] > 1e-3:
        wall_cx = moments["m10"] / moments["m00"]
        wall_cy = moments["m01"] / moments["m00"]
        # Vector from image centre TO the wall centroid
        dx = wall_cx - cx
        dy = wall_cy - cy
        dist_px = np.hypot(dx, dy) + 1e-6
        # Normal points AWAY from wall (we want the scope to move this way)
        normal = np.array([-dx / dist_px, -dy / dist_px])
    else:
        normal = np.array([0.0, 0.0])

    # Map dark_ratio → metric distance heuristic (calibrate per setup)
    # 0.0 dark → 0.5 m away;  1.0 dark → ~0 m (touching wall)
    dist = max(0.01, 0.5 * (1.0 - dark_ratio))
    margin = np.clip(1.0 - dark_ratio, 0.0, 1.0)

    return dist, normal, margin


# ===========================================================================
# 1. COLLISION GUARD  (Manual mode safety)
# ===========================================================================

class CollisionGuard:
    """
    Wraps every manual joint command and either passes it through unchanged
    (when the tip is far from walls) or modifies it to prevent wall contact.

    Design (Chapter 11 §11.3.3 feedforward + PI):
    ─────────────────────────────────────────────
    The guard treats the end-effector tip position as the controlled output.
    When the vision pipeline reports that the tip is inside GUARD_WALL_THRESHOLD
    of a wall, it computes a corrective body-frame twist using a PI controller
    and projects that twist back to joint-space via the pseudo-inverse Jacobian.

    The user's commanded Δθ is then damped and the corrective Δθ is added so
    that the net motion steers the tip away from the wall while honouring the
    user's *intent* as much as possible (the component of the user's command
    that is tangent to the wall is preserved).
    """

    def __init__(self, kin: BronchoKinematics):
        self.kin = kin
        # PI integrator for wall-avoidance error
        self._integral = np.zeros(2)  # lateral (X, Y) in task space

    def reset(self):
        self._integral[:] = 0.0

    def filter_command(
        self,
        state: RobotState,
        delta_insertion: float,
        delta_proximal: float,
        delta_mid: float,
        delta_distal: float,
        frame=None,
    ) -> RobotState:
        """
        Apply the guard filter and return a safe new RobotState.

        Parameters
        ----------
        state            : current joint state
        delta_*          : the INCREMENTAL changes the user requested
        frame            : latest BGR camera frame (may be None)

        Returns
        -------
        new_state : RobotState with safe joint values
        """
        # ── Forward kinematics: where is the tip right now? ──────────────────
        thetas = state.as_list
        T1 = self.kin.matrix_exp_6(self.kin.S1, thetas[0])
        T2 = self.kin.matrix_exp_6(self.kin.S2, thetas[1])
        T3 = self.kin.matrix_exp_6(self.kin.S3, thetas[2])
        T4 = self.kin.matrix_exp_6(self.kin.S4, thetas[3])
        T_sb = np.dot(np.dot(np.dot(T1, T2), T3), T4).dot(self.kin.M)
        tip_xy = T_sb[:2, 3]

        # ── Sense proximity ──────────────────────────────────────────────────
        dist, normal_2d, margin = estimate_wall_proximity(frame, tip_xy)

        # ── Compute user-requested delta vector ──────────────────────────────
        d_theta = np.array([delta_insertion, delta_proximal,
                            delta_mid,       delta_distal], dtype=float)

        if dist >= GUARD_WALL_THRESHOLD:
            # Safe zone – pass through unmodified
            self._integral[:] = 0.0
            new = RobotState(
                state.insertion + d_theta[0],
                state.proximal  + d_theta[1],
                state.mid       + d_theta[2],
                state.distal    + d_theta[3],
            )
            return new.clipped()

        # ── Inside danger zone ───────────────────────────────────────────────
        # Lateral error: how far inside the threshold are we?
        error_xy = normal_2d * (GUARD_WALL_THRESHOLD - dist)

        # PI update (§11.3.1 P-control + integral)
        self._integral += error_xy
        corrective_xy = GUARD_KP * error_xy + GUARD_KI * self._integral

        # Build a 6-D body twist from the 2-D corrective vector
        # (we only push laterally; no z-correction to preserve insertion intent)
        V_corrective = np.array([0.0, 0.0, 0.0,
                                  corrective_xy[0],
                                  corrective_xy[1],
                                  0.0])

        # Map corrective twist → joint corrections via pseudo-inverse Jacobian
        Js = self.kin.jacobian_space(thetas)
        T_final = T_sb
        Jb = self.kin.jacobian_body(Js, T_final)
        Jb_pinv = np.linalg.pinv(Jb)
        d_theta_correction = np.dot(Jb_pinv, V_corrective)
        d_theta_correction = np.clip(
            d_theta_correction, -GUARD_MAX_CORRECTION, GUARD_MAX_CORRECTION
        )

        # Damp the user's command component that drives INTO the wall
        # (preserve the component that is safe / parallel to the wall)
        user_tip_motion = np.dot(Jb[3:5, :], d_theta)  # XY velocity
        into_wall = np.dot(user_tip_motion, -normal_2d)
        if into_wall > 0:
            # User is pushing toward the wall → damp that component
            d_theta_safe = d_theta * GUARD_DAMP
        else:
            d_theta_safe = d_theta  # moving away already – allow freely

        combined = d_theta_safe + d_theta_correction

        new = RobotState(
            state.insertion + combined[0],
            state.proximal  + combined[1],
            state.mid       + combined[2],
            state.distal    + combined[3],
        )
        return new.clipped()


# ===========================================================================
# 2. AUTO PILOT  (Autonomous navigation)
# ===========================================================================

class AutoPilot:
    """
    Task-space feedforward + PI controller for autonomous bronchoscope
    navigation (Chapter 11 §11.3.3 and §11.4.3).

    Behaviour
    ─────────
    1. ADVANCE   – Insert the scope at AUTO_INSERT_SPEED per tick.
    2. SENSE     – Read wall proximity from the vision pipeline.
    3. CORRECT   – If the tip is inside AUTO_WALL_ZONE, compute a repulsive
                   body twist and map it to joint corrections via J†.
    4. GOAL      – If the tip X-coordinate exceeds goal_x, stop and signal
                   completion.

    The PI lateral controller follows §11.3.1:
        V_lateral = Kp * e + Ki * ∫e dt

    where e is the signed lateral offset required to centre the scope in
    the detected lumen.
    """

    def __init__(self, kin: BronchoKinematics, goal_x: float = 1.4):
        self.kin    = kin
        self.goal_x = goal_x
        self._integral = np.zeros(2)
        self._done  = False
        self._tick  = 0

    def reset(self, goal_x: float = 1.4):
        self._integral[:] = 0.0
        self._done  = False
        self._tick  = 0
        self.goal_x = goal_x

    @property
    def is_done(self):
        return self._done

    def step(self, state: RobotState, frame=None) -> RobotState:
        """
        Compute the next safe joint state autonomously.

        Parameters
        ----------
        state  : current joint state
        frame  : latest BGR camera frame (may be None)

        Returns
        -------
        new_state : RobotState for the next tick
        """
        if self._done:
            return state

        self._tick += 1

        # ── Forward kinematics ───────────────────────────────────────────────
        thetas = state.as_list
        T1 = self.kin.matrix_exp_6(self.kin.S1, thetas[0])
        T2 = self.kin.matrix_exp_6(self.kin.S2, thetas[1])
        T3 = self.kin.matrix_exp_6(self.kin.S3, thetas[2])
        T4 = self.kin.matrix_exp_6(self.kin.S4, thetas[3])
        T_sb = np.dot(np.dot(np.dot(T1, T2), T3), T4).dot(self.kin.M)
        tip   = T_sb[:3, 3]
        tip_xy = tip[:2]

        # ── Goal check ───────────────────────────────────────────────────────
        if tip[0] >= self.goal_x:
            self._done = True
            return state

        # ── Sense proximity ──────────────────────────────────────────────────
        dist, normal_2d, margin = estimate_wall_proximity(frame, tip_xy)

        # ── 1. Feedforward insertion (§11.4.1.2) ────────────────────────────
        # Simply advance along the scope's insertion axis.
        new_insertion = state.insertion + AUTO_INSERT_SPEED

        # ── 2. Lateral PI correction (§11.3.3 task-space PI) ────────────────
        if dist < AUTO_WALL_ZONE:
            # Error = desired clearance – actual clearance, projected laterally
            error_xy = normal_2d * (AUTO_WALL_ZONE - dist) * AUTO_REPULSE_GAIN
        else:
            error_xy = np.zeros(2)

        # PI update
        self._integral += error_xy
        lateral_correction = AUTO_BEND_KP * error_xy + AUTO_BEND_KI * self._integral

        # Clamp integrator to prevent wind-up (§11.4.1.1 anti-windup)
        self._integral = np.clip(self._integral, -2.0, 2.0)

        # ── 3. Map lateral twist → joint corrections ─────────────────────────
        V_b = np.array([0.0, 0.0, 0.0,
                        lateral_correction[0],
                        lateral_correction[1],
                        0.0])

        Js = self.kin.jacobian_space(thetas)
        Jb = self.kin.jacobian_body(Js, T_sb)

        # Check manipulability – avoid commands at singularities
        ellips = self.kin.ellipsoid_analysis(Js)
        is_singular = (ellips['linear']['mu1'] == float('inf'))

        if not is_singular:
            Jb_pinv = np.linalg.pinv(Jb)
            d_theta  = np.dot(Jb_pinv, V_b)
            d_theta  = np.clip(d_theta, -0.04, 0.04)
        else:
            d_theta = np.zeros(4)

        # ── 4. Smooth sinusoidal tip oscillation when no obstacle nearby ─────
        # Mimics an experienced endoscopist's gentle probe motion (keeps the
        # lumen centred without relying solely on vision).
        if dist >= AUTO_WALL_ZONE:
            osc_freq   = 0.05   # cycles per tick
            osc_amp    = 0.015  # radians
            oscillation = osc_amp * np.sin(2 * np.pi * osc_freq * self._tick)
            d_theta[3] += oscillation   # wiggle distal tip

        # ── 5. Assemble new state ────────────────────────────────────────────
        new = RobotState(
            insertion = new_insertion,
            proximal  = state.proximal  + d_theta[1],
            mid       = state.mid       + d_theta[2],
            distal    = state.distal    + d_theta[3],
        )
        return new.clipped()


# ===========================================================================
# 3. STATUS / HUD data  (helpers for the dashboard overlay)
# ===========================================================================

def guard_status_text(dist: float, margin: float) -> str:
    """One-line status string for the HUD."""
    if dist < GUARD_WALL_THRESHOLD * 0.5:
        return f"⚠ COLLISION GUARD ACTIVE  dist={dist:.3f}m"
    elif dist < GUARD_WALL_THRESHOLD:
        return f"! GUARD WATCHING  dist={dist:.3f}m  margin={margin:.2f}"
    return f"GUARD OK  dist={dist:.3f}m"


def auto_status_text(pilot: AutoPilot, state: RobotState) -> str:
    """One-line status string for AUTO mode HUD."""
    if pilot.is_done:
        return "✓ AUTO: GOAL REACHED"
    return f"AUTO  tick={pilot._tick}  ins={state.insertion:.3f}m"

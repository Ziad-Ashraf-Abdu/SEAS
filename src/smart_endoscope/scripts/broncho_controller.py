#!/usr/bin/env python3
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
# Tunable parameters
# ---------------------------------------------------------------------------
GUARD_WALL_THRESHOLD   = 0.08   
GUARD_KP               = 1.20   
GUARD_KI               = 0.35   
GUARD_DAMP             = 0.80   
GUARD_MAX_CORRECTION   = 0.25   

AUTO_INSERT_SPEED      = 0.008  
AUTO_BEND_KP           = 1.80   
AUTO_BEND_KI           = 0.40   
AUTO_MAX_BEND          = 1.45   
AUTO_GOAL_THRESHOLD    = 0.02   
AUTO_WALL_ZONE         = 0.12   
AUTO_REPULSE_GAIN      = 2.50   

# Vertical rail step per key-press (metres)
VERTICAL_STEP = 0.01   # 1 cm per key event

# ---------------------------------------------------------------------------
# Robot State (Now 8-DOF: vertical rail + 7 arm joints)
# ---------------------------------------------------------------------------
class RobotState:
    def __init__(self, vertical=0.0, insertion=0.0, prox_yaw=0.0, prox_pitch=0.0,
                 mid_yaw=0.0, mid_pitch=0.0, dist_yaw=0.0, dist_roll=0.0):
        self.vertical   = vertical      # world-Z translation of entire arm (m)
        self.insertion  = insertion
        self.prox_yaw   = prox_yaw
        self.prox_pitch = prox_pitch
        self.mid_yaw    = mid_yaw
        self.mid_pitch  = mid_pitch
        self.dist_yaw   = dist_yaw
        self.dist_roll  = dist_roll

    @property
    def as_list(self):
        # Order MUST match controllers.yaml joint order:
        # vertical_joint, insertion_joint, proximal_yaw, proximal_pitch,
        # mid_yaw, mid_pitch, distal_yaw, distal_roll
        return [self.vertical, self.insertion, self.prox_yaw, self.prox_pitch,
                self.mid_yaw, self.mid_pitch, self.dist_yaw, self.dist_roll]

    def clipped(self):
        return RobotState(
            vertical   = np.clip(self.vertical,   -0.30,  0.30),
            insertion  = np.clip(self.insertion,  -0.50, 30.0),
            prox_yaw   = np.clip(self.prox_yaw,   -0.60,  0.60),
            prox_pitch = np.clip(self.prox_pitch, -0.60,  0.60),
            mid_yaw    = np.clip(self.mid_yaw,    -1.00,  1.00),
            mid_pitch  = np.clip(self.mid_pitch,  -1.00,  1.00),
            dist_yaw   = np.clip(self.dist_yaw,   -1.57,  1.57),
            dist_roll  = np.clip(self.dist_roll,  -3.14,  3.14),
        )

# ---------------------------------------------------------------------------
# Proximity reading
# ---------------------------------------------------------------------------
def estimate_wall_proximity(frame, tip_pos_2d):
    import cv2
    if frame is None: return 0.5, np.array([0.0, 1.0]), 1.0

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    roi_size = min(h, w) // 3
    x0, x1 = max(0, cx - roi_size), min(w, cx + roi_size)
    y0, y1 = max(0, cy - roi_size), min(h, cy + roi_size)
    roi = dark_mask[y0:y1, x0:x1]

    dark_ratio = roi.mean() / 255.0
    moments = cv2.moments(dark_mask)
    if moments["m00"] > 1e-3:
        wall_cx = moments["m10"] / moments["m00"]
        wall_cy = moments["m01"] / moments["m00"]
        dx, dy = wall_cx - cx, wall_cy - cy
        dist_px = np.hypot(dx, dy) + 1e-6
        normal = np.array([-dx / dist_px, -dy / dist_px])
    else:
        normal = np.array([0.0, 0.0])

    dist = max(0.01, 0.5 * (1.0 - dark_ratio))
    margin = np.clip(1.0 - dark_ratio, 0.0, 1.0)
    return dist, normal, margin

# ===========================================================================
# 1. COLLISION GUARD
# ===========================================================================
class CollisionGuard:
    def __init__(self, kin: BronchoKinematics):
        self.kin = kin
        self._integral = np.zeros(2)

    def reset(self):
        self._integral[:] = 0.0

    def filter_command(self, state: RobotState, d_theta: list, frame=None) -> RobotState:
        # vertical_joint (index 0) is independent of the arm kinematics —
        # slice it off before passing to BronchoKinematics which expects 7 values
        all_vals = state.as_list                   # length 8
        arm_thetas = all_vals[1:]                  # length 7 — arm joints only
        d_all = np.array(d_theta, dtype=float)     # length 8
        d_arm = d_all[1:]                          # length 7

        T_sb = self.kin.forward_kinematics_space(arm_thetas)
        tip_xy = T_sb[:2, 3]

        dist, normal_2d, margin = estimate_wall_proximity(frame, tip_xy)

        if dist >= GUARD_WALL_THRESHOLD:
            self._integral[:] = 0.0
            new_vals = np.array(all_vals) + d_all
            return RobotState(*new_vals).clipped()

        error_xy = normal_2d * (GUARD_WALL_THRESHOLD - dist)
        self._integral += error_xy
        corrective_xy = GUARD_KP * error_xy + GUARD_KI * self._integral

        V_corrective = np.array([0.0, 0.0, 0.0, corrective_xy[0], corrective_xy[1], 0.0])

        Js = self.kin.jacobian_space(arm_thetas)
        Jb = self.kin.jacobian_body(Js, T_sb)
        Jb_pinv = np.linalg.pinv(Jb)

        d_arm_correction = np.clip(
            np.dot(Jb_pinv, V_corrective), -GUARD_MAX_CORRECTION, GUARD_MAX_CORRECTION
        )

        user_tip_motion = np.dot(Jb[3:5, :], d_arm)
        into_wall = np.dot(user_tip_motion, -normal_2d)
        d_arm_safe = d_arm * GUARD_DAMP if into_wall > 0 else d_arm

        # Combine: vertical passes through unmodified; arm joints get safety filter
        new_arm = np.array(arm_thetas) + d_arm_safe + d_arm_correction
        new_vertical = all_vals[0] + d_all[0]
        return RobotState(new_vertical, *new_arm).clipped()

# ===========================================================================
# 2. AUTO PILOT
# ===========================================================================
class AutoPilot:
    def __init__(self, kin: BronchoKinematics, goal_x: float = 1.4):
        self.kin, self.goal_x = kin, goal_x
        self._integral = np.zeros(2)
        self._done, self._tick = False, 0

    def reset(self, goal_x: float = 1.4):
        self._integral[:], self._done, self._tick, self.goal_x = 0.0, False, 0, goal_x

    @property
    def is_done(self): return self._done

    def step(self, state: RobotState, frame=None) -> RobotState:
        if self._done: return state
        self._tick += 1

        all_vals   = state.as_list      # length 8
        arm_thetas = all_vals[1:]       # length 7 — arm joints only

        T_sb = self.kin.forward_kinematics_space(arm_thetas)
        tip = T_sb[:3, 3]

        distance_to_goal = self.goal_x - tip[0]

        if abs(distance_to_goal) <= AUTO_GOAL_THRESHOLD:
            self._done = True
            return state

        direction = np.sign(distance_to_goal)

        dist, normal_2d, _ = estimate_wall_proximity(frame, tip[:2])

        new_insertion = state.insertion + (direction * AUTO_INSERT_SPEED)

        error_xy = normal_2d * (AUTO_WALL_ZONE - dist) * AUTO_REPULSE_GAIN if dist < AUTO_WALL_ZONE else np.zeros(2)
        self._integral = np.clip(self._integral + error_xy, -2.0, 2.0)
        lateral_correction = AUTO_BEND_KP * error_xy + AUTO_BEND_KI * self._integral

        V_b = np.array([0.0, 0.0, 0.0, lateral_correction[0], lateral_correction[1], 0.0])
        Js = self.kin.jacobian_space(arm_thetas)
        Jb = self.kin.jacobian_body(Js, T_sb)

        ellips = self.kin.ellipsoid_analysis(Js)
        if ellips['linear']['mu1'] != float('inf'):
            d_arm = np.clip(np.dot(np.linalg.pinv(Jb), V_b), -0.04, 0.04)
        else:
            d_arm = np.zeros(7)

        if dist >= AUTO_WALL_ZONE:
            d_arm[5] += 0.015 * np.sin(2 * np.pi * 0.05 * self._tick)

        new_arm = np.array(arm_thetas) + d_arm
        new_arm[0] = new_insertion   # override insertion with forward-drive value
        # vertical_joint is untouched during autopilot — keep current value
        return RobotState(state.vertical, *new_arm).clipped()

# ===========================================================================
# 3. STATUS
# ===========================================================================
def guard_status_text(dist: float, margin: float) -> str:
    if dist < GUARD_WALL_THRESHOLD * 0.5: return f"⚠ COLLISION GUARD ACTIVE  dist={dist:.3f}m"
    elif dist < GUARD_WALL_THRESHOLD: return f"! GUARD WATCHING  dist={dist:.3f}m  margin={margin:.2f}"
    return f"GUARD OK  dist={dist:.3f}m"

def auto_status_text(pilot: AutoPilot, state: RobotState) -> str:
    if pilot.is_done: return "✓ AUTO: GOAL REACHED"
    return f"AUTO  tick={pilot._tick}  ins={state.insertion:.3f}m"
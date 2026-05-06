"""
broncho_kinematics.py
=====================
Math engine for the Smart Bronchoscope.

Chapter coverage
────────────────
Ch. 3  Rigid-Body Motions
       skew_symmetric, trans_inv, matrix_exp_6, matrix_log_3, matrix_log_6
       adjoint_matrix, create_wrench

Ch. 4  Forward Kinematics (Product of Exponentials)
       forward_kinematics_space  (PoE, space-frame screw axes)
       forward_kinematics_body   (PoE, body-frame screw axes)

Ch. 5  Velocity Kinematics & Statics
       jacobian_space, jacobian_body, static_torques, ellipsoid_analysis

Ch. 6  Inverse Kinematics
       ik_body                   (Newton-Raphson numerical IK)
       inverse_velocity_kinematics

Ch. 8  Dynamics of Open Chains  ← NEW
       ad                        (Lie bracket / adjoint action on twists)
       inverse_dynamics          (recursive Newton-Euler)
       mass_matrix               (M(θ) via Newton-Euler)
       velocity_quadratic_forces (Coriolis + centripetal  c(θ,θ̇))
       gravity_forces            (g(θ))
       forward_dynamics          (θ̈ given τ)

Ch. 9  Trajectory Generation    ← NEW
       cubic_time_scaling        (3rd-order polynomial time scaling)
       quintic_time_scaling      (5th-order polynomial time scaling)
       trapezoidal_time_scaling  (bang-coast-bang profile)
       joint_trajectory_cubic    (full θ(t) for point-to-point)
       screw_trajectory          (straight-line SE(3) screw path)
       cartesian_trajectory      (decoupled rotation + translation path)
       via_point_trajectory      (piecewise-cubic through via points)

RULE: Every kinematic/dynamic equation lives HERE, nowhere else.
      broncho_controller.py only calls these methods.
"""

import numpy as np


class BronchoKinematics:

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(self, L1=0.5, L2=0.5, L3=0.5):
        """
        Initialise the 4-DOF kinematic chain.

        Joint layout (all in the XY-plane):
          Joint 1 – Prismatic along X     (insertion)
          Joint 2 – Revolute about Z at x = L1
          Joint 3 – Revolute about Z at x = L1+L2
          Joint 4 – Revolute about Z at x = L1+L2+L3

        L1, L2, L3 should match the URDF link lengths (metres).
        """
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # Home configuration M (end-effector when θ = 0)
        self.M = np.array([
            [1, 0, 0, L1 + L2 + L3],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # ── Chapter 3: Screw Axes in the base/space frame ──────────────────
        # S = [ω_x, ω_y, ω_z, v_x, v_y, v_z]

        # Joint 1: Prismatic along X
        self.S1 = np.array([0, 0, 0,  1, 0, 0], dtype=float)

        # Joint 2: Revolute about Z, axis through x = L1
        self.S2 = np.array([0, 0, 1,  0, -L1, 0], dtype=float)

        # Joint 3: Revolute about Z, axis through x = L1+L2
        self.S3 = np.array([0, 0, 1,  0, -(L1 + L2), 0], dtype=float)

        # Joint 4: Revolute about Z, axis through x = L1+L2+L3
        self.S4 = np.array([0, 0, 1,  0, -(L1 + L2 + L3), 0], dtype=float)

        self.S_list = [self.S1, self.S2, self.S3, self.S4]

        # Placeholders for future 6-DOF upgrade
        self.S5 = None
        self.S6 = None

        # ── Chapter 8: Spatial inertia matrices ─────────────────────────────
        # Glist[i] is the 6×6 spatial inertia of link i in its own CoM frame.
        # These are placeholder values – replace with real URDF inertials.
        #
        # Spatial inertia G = diag(Ixx, Iyy, Izz, m, m, m)  (diagonal approx)
        #
        # Link masses (kg) and scalar inertias (kg m²) – placeholders
        link_masses   = [0.08, 0.06, 0.04, 0.02]
        link_inertias = [5e-5, 3e-5, 1e-5, 5e-6]  # I_zz dominant (planar)

        self.Glist = []
        for m, I in zip(link_masses, link_inertias):
            G = np.diag([I, I, I, m, m, m])
            self.Glist.append(G)

        # Mlist[i] = config of {i-1} in {i} when θ=0  (for Newton-Euler)
        # Each frame is placed at the link CoM; spacing ≈ Li
        lengths = [L1, L2, L3, 0.0]  # last is end-effector (zero offset)
        self.Mlist = []
        for l in lengths:
            Mi = np.eye(4)
            Mi[0, 3] = l
            self.Mlist.append(Mi)

        # Gravity vector in the base frame (bronchoscope is mostly horizontal)
        self.gravity = np.array([0, 0, -9.81])

    # =========================================================================
    # Chapter 3 – Rigid-Body Motions
    # =========================================================================

    @staticmethod
    def skew_symmetric(vec3):
        """3-vector → 3×3 skew-symmetric matrix [ω]."""
        return np.array([
            [0,        -vec3[2],  vec3[1]],
            [vec3[2],   0,       -vec3[0]],
            [-vec3[1],  vec3[0],  0      ],
        ])

    def trans_inv(self, T):
        """
        Ch.3 – Efficient inverse of a homogeneous transformation matrix.
        Avoids np.linalg.inv; exploits the SO(3) structure of R.
        """
        R = T[:3, :3]
        p = T[:3,  3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3,  3] = -R.T @ p
        return T_inv

    def matrix_exp_6(self, S, theta):
        """
        Ch.3 – Matrix exponential e^{[S]θ} ∈ SE(3).
        Works for both prismatic (ω=0) and revolute joints.
        """
        omega = S[:3]
        v     = S[3:]
        T     = np.eye(4)

        if np.linalg.norm(omega) < 1e-6:          # Prismatic
            T[:3, 3] = v * theta
            return T

        omega_skew = self.skew_symmetric(omega)
        # Rodrigues: R = I + sin θ [ω] + (1−cos θ)[ω]²
        R = (np.eye(3)
             + np.sin(theta) * omega_skew
             + (1 - np.cos(theta)) * omega_skew @ omega_skew)
        # G(θ) = Iθ + (1−cos θ)[ω] + (θ−sin θ)[ω]²
        G = (np.eye(3) * theta
             + (1 - np.cos(theta)) * omega_skew
             + (theta - np.sin(theta)) * omega_skew @ omega_skew)

        T[:3, :3] = R
        T[:3,  3] = G @ v
        return T

    def adjoint_matrix(self, T):
        """
        Ch.3 – 6×6 Adjoint representation [AdT].
        Maps twists and (dually) wrenches between frames.
        """
        R = T[:3, :3]
        p = T[:3,  3]
        p_skew = self.skew_symmetric(p)

        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, :3] = p_skew @ R
        AdT[3:, 3:] = R
        return AdT

    def create_wrench(self, forces, torques):
        """Ch.3 – Pack (f, m) into a 6-D wrench F = [m, f]^T."""
        return np.concatenate((torques, forces))

    def matrix_log_3(self, R):
        """Ch.3 – Matrix logarithm of a rotation matrix R ∈ SO(3)."""
        acos_input = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(acos_input)

        if np.isclose(theta, 0.0):
            return np.zeros((3, 3))

        if np.isclose(theta, np.pi):
            # Singularity (trace = -1): pick safe column
            if not np.isclose(1 + R[2, 2], 0.0):
                omega = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif not np.isclose(1 + R[1, 1], 0.0):
                omega = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else:
                omega = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            return self.skew_symmetric(omega) * np.pi

        return (theta / (2 * np.sin(theta))) * (R - R.T)

    def matrix_log_6(self, T):
        """
        Ch.3 – Matrix logarithm of T ∈ SE(3).
        Returns a 6-D twist vector [ω·θ, v·θ].
        """
        R = T[:3, :3]
        p = T[:3,  3]

        omega_skew = self.matrix_log_3(R)

        if np.allclose(omega_skew, 0):
            v_theta = p
        else:
            theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
            G_inv = (np.eye(3)
                     - 0.5 * omega_skew
                     + (1.0 / theta - 0.5 / np.tan(theta / 2.0))
                     * (omega_skew @ omega_skew) / theta)
            v_theta = G_inv @ p

        omega_theta = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        return np.concatenate((omega_theta, v_theta))

    # =========================================================================
    # Chapter 4 – Forward Kinematics (Product of Exponentials)
    # =========================================================================

    def forward_kinematics_space(self, theta_list):
        """
        Ch.4 §4.1.1 – PoE formula using space-frame screw axes.
        T(θ) = e^{[S1]θ1} · e^{[S2]θ2} · … · e^{[Sn]θn} · M
        """
        T = np.eye(4)
        for S, theta in zip(self.S_list, theta_list):
            T = T @ self.matrix_exp_6(S, theta)
        return T @ self.M

    def forward_kinematics_body(self, B_list, theta_list):
        """
        Ch.4 §4.1.3 – PoE formula using body-frame screw axes.
        T(θ) = M · e^{[B1]θ1} · … · e^{[Bn]θn}
        """
        T = np.copy(self.M)
        for B, theta in zip(B_list, theta_list):
            T = T @ self.matrix_exp_6(B, theta)
        return T

    # =========================================================================
    # Chapter 5 – Velocity Kinematics & Statics
    # =========================================================================

    def jacobian_space(self, theta_list):
        """
        Ch.5 §5.1.1 – 6×4 Space Jacobian Js(θ).
        Column i = Ad_{T_{0→i-1}} · Si
        """
        Js = np.zeros((6, 4))
        Js[:, 0] = self.S1

        T1   = self.matrix_exp_6(self.S1, theta_list[0])
        Js[:, 1] = self.adjoint_matrix(T1) @ self.S2

        T2   = self.matrix_exp_6(self.S2, theta_list[1])
        T12  = T1 @ T2
        Js[:, 2] = self.adjoint_matrix(T12) @ self.S3

        T3   = self.matrix_exp_6(self.S3, theta_list[2])
        T123 = T12 @ T3
        Js[:, 3] = self.adjoint_matrix(T123) @ self.S4

        return Js

    def jacobian_body(self, Js, T_final):
        """
        Ch.5 §5.1.2 – Body Jacobian Jb(θ) = [Ad_{T^{-1}}] Js.
        T_final is the end-effector transformation T_sb.
        """
        return self.adjoint_matrix(self.trans_inv(T_final)) @ Js

    def static_torques(self, J, F):
        """
        Ch.5 §5.2 – Joint torques to resist wrench F:  τ = Jᵀ F.
        J can be Js (F in space frame) or Jb (F in body frame).
        """
        return J.T @ F

    def ellipsoid_analysis(self, J):
        """
        Ch.5 §5.4 – Manipulability ellipsoid measures μ₁, μ₂, μ₃
        for both angular (top 3 rows) and linear (bottom 3 rows) Jacobian.
        Returns dict with keys 'angular' and 'linear', each containing
        {'mu1': condition_number_sqrt, 'mu2': condition_number, 'mu3': volume}.
        """
        J_omega = J[:3, :]
        J_v     = J[3:, :]

        def calculate_mu(A_mat):
            eigs = np.clip(np.linalg.eigvalsh(A_mat), 0.0, None)
            eigs = np.sort(eigs)
            lmin, lmax = eigs[0], eigs[-1]
            mu3 = np.sqrt(max(0.0, np.linalg.det(A_mat)))
            if np.isclose(lmin, 0.0, atol=1e-6):
                return float('inf'), float('inf'), mu3
            return np.sqrt(lmax / lmin), lmax / lmin, mu3

        mu1w, mu2w, mu3w = calculate_mu(J_omega @ J_omega.T)
        mu1v, mu2v, mu3v = calculate_mu(J_v     @ J_v.T)

        return {
            'angular': {'mu1': mu1w, 'mu2': mu2w, 'mu3': mu3w},
            'linear':  {'mu1': mu1v, 'mu2': mu2v, 'mu3': mu3v},
        }

    # =========================================================================
    # Chapter 6 – Inverse Kinematics
    # =========================================================================

    def ik_body(self, T_sd, theta_guess,
                e_omega=0.001, e_v=0.0001, max_iter=50):
        """
        Ch.6 §6.2 – Numerical IK via Newton-Raphson (body Jacobian).
        Returns (theta, converged).
        """
        theta = np.array(theta_guess, dtype=float)

        for _ in range(max_iter):
            T_sb = self.forward_kinematics_space(theta)
            T_bd = self.trans_inv(T_sb) @ T_sd
            V_b  = self.matrix_log_6(T_bd)

            if (np.linalg.norm(V_b[:3]) < e_omega
                    and np.linalg.norm(V_b[3:]) < e_v):
                return theta, True

            Js   = self.jacobian_space(theta)
            Jb   = self.jacobian_body(Js, T_sb)
            theta += np.linalg.pinv(Jb) @ V_b

        return theta, False

    def inverse_velocity_kinematics(self, J, V_d):
        """
        Ch.6 §6.3 – Joint velocities for desired twist V_d:
        θ̇ = J† V_d.
        """
        return np.linalg.pinv(J) @ V_d

    # =========================================================================
    # Chapter 8 – Dynamics of Open Chains   (NEW)
    # =========================================================================

    @staticmethod
    def ad(V):
        """
        Ch.8 – 6×6 matrix [adV] (Lie bracket / adjoint action on twists).

        For V = (ω, v):
            [adV] = | [ω]   0  |
                    | [v]  [ω] |

        Used in Newton-Euler forward/backward recursions.
        """
        omega_skew = BronchoKinematics.skew_symmetric.__func__(None, V[:3])
        v_skew     = BronchoKinematics.skew_symmetric.__func__(None, V[3:])

        adV = np.zeros((6, 6))
        adV[:3, :3] = omega_skew
        adV[3:, :3] = v_skew
        adV[3:, 3:] = omega_skew
        return adV

    def inverse_dynamics(self, theta_list, dtheta_list, ddtheta_list,
                         g=None, F_tip=None):
        """
        Ch.8 §8.3 – Recursive Newton-Euler inverse dynamics.

        Computes the joint torques/forces τ required to produce the given
        motion (θ, θ̇, θ̈) against gravity and an optional tip wrench.

        Parameters
        ----------
        theta_list   : (4,) joint positions
        dtheta_list  : (4,) joint velocities
        ddtheta_list : (4,) joint accelerations
        g            : (3,) gravity vector in base frame (default self.gravity)
        F_tip        : (6,) wrench applied BY end-effector ON environment
                       expressed in the end-effector frame (default zeros)

        Returns
        -------
        tau : (4,) joint torques/forces
        """
        if g     is None: g     = self.gravity
        if F_tip is None: F_tip = np.zeros(6)

        n       = len(theta_list)
        Mi_list = self.Mlist
        Gi_list = self.Glist
        Si_list = self.S_list

        # Initialise base twist as gravity treated as acceleration
        V0    = np.zeros(6)
        Vd0   = np.zeros(6)
        Vd0[3:] = -g          # gravity → upward fictitious acceleration

        # ── Forward pass: propagate twists and accelerations ─────────────
        V_list  = [V0]
        Vd_list = [Vd0]
        T_list  = []          # Ti,i-1 for each joint

        for i in range(n):
            Ai    = Si_list[i]
            Mi    = Mi_list[i]
            theta = theta_list[i]
            dth   = dtheta_list[i]
            ddth  = ddtheta_list[i]

            Ti_im1 = self.matrix_exp_6(-Ai, theta) @ Mi  # T_{i,i-1}
            T_list.append(Ti_im1)

            AdTi   = self.adjoint_matrix(Ti_im1)
            Vi     = AdTi @ V_list[-1]  + Ai * dth
            Vdi    = AdTi @ Vd_list[-1] + self.ad(Vi) @ Ai * dth + Ai * ddth
            V_list.append(Vi)
            Vd_list.append(Vdi)

        # ── Backward pass: propagate wrenches ────────────────────────────
        F_list = [F_tip]
        tau    = np.zeros(n)

        for i in range(n - 1, -1, -1):
            Gi = Gi_list[i]
            Vi = V_list[i + 1]
            Vdi= Vd_list[i + 1]
            Ti = T_list[i]

            # Wrench at link i+1 expressed in frame i
            if i < n - 1:
                T_next = T_list[i + 1]
                AdT_next = self.adjoint_matrix(T_next)
                Fi = (AdT_next.T @ F_list[-1]
                      + Gi @ Vdi
                      - self.ad(Vi).T @ Gi @ Vi)
            else:
                # Last link: wrench from tip
                AdT_tip = self.adjoint_matrix(T_list[-1])
                Fi = (AdT_tip.T @ F_tip
                      + Gi @ Vdi
                      - self.ad(Vi).T @ Gi @ Vi)

            F_list.append(Fi)
            tau[i] = float(Si_list[i] @ Fi)

        return tau

    def mass_matrix(self, theta_list):
        """
        Ch.8 §8.4 – Joint-space mass matrix M(θ), shape (4,4).

        Computed by calling inverse_dynamics n times with unit accelerations
        and zero velocity/gravity (standard textbook approach).
        """
        n   = len(theta_list)
        M   = np.zeros((n, n))
        g0  = np.zeros(3)      # zero gravity for mass matrix

        for i in range(n):
            ddtheta       = np.zeros(n)
            ddtheta[i]    = 1.0
            tau = self.inverse_dynamics(
                theta_list,
                np.zeros(n),
                ddtheta,
                g=g0,
                F_tip=np.zeros(6),
            )
            M[:, i] = tau

        return M

    def velocity_quadratic_forces(self, theta_list, dtheta_list):
        """
        Ch.8 §8.4 – Coriolis + centripetal forces  c(θ, θ̇).
        Computed as τ = ID(θ, θ̇, 0, g=0) − g(θ).
        """
        tau_with_vel = self.inverse_dynamics(
            theta_list, dtheta_list, np.zeros(len(theta_list)),
            g=np.zeros(3),
        )
        return tau_with_vel

    def gravity_forces(self, theta_list):
        """
        Ch.8 §8.4 – Gravity torques  g(θ).
        Computed as τ = ID(θ, 0, 0, g=self.gravity).
        """
        return self.inverse_dynamics(
            theta_list,
            np.zeros(len(theta_list)),
            np.zeros(len(theta_list)),
        )

    def forward_dynamics(self, theta_list, dtheta_list, tau,
                         g=None, F_tip=None):
        """
        Ch.8 §8.5 – Forward dynamics: solve for θ̈ given τ.

        θ̈ = M(θ)⁻¹ (τ − c(θ,θ̇) − g(θ) − Jᵀ F_tip)

        Returns
        -------
        ddtheta : (4,) joint accelerations
        """
        if g     is None: g     = self.gravity
        if F_tip is None: F_tip = np.zeros(6)

        n   = len(theta_list)
        M   = self.mass_matrix(theta_list)
        c   = self.velocity_quadratic_forces(theta_list, dtheta_list)
        grav= self.gravity_forces(theta_list)

        # End-effector force contribution
        T_final = self.forward_kinematics_space(theta_list)
        Js      = self.jacobian_space(theta_list)
        Jb      = self.jacobian_body(Js, T_final)
        tau_tip = Jb.T @ F_tip

        rhs  = tau - c - grav - tau_tip
        # Use lstsq for robustness near singularities
        ddtheta, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return ddtheta

    # =========================================================================
    # Chapter 9 – Trajectory Generation   (NEW)
    # =========================================================================

    # ── Time-scaling functions ────────────────────────────────────────────────

    @staticmethod
    def cubic_time_scaling(T, t):
        """
        Ch.9 §9.2.2.1 – Third-order polynomial time scaling.

        s(t)  = 3(t/T)² − 2(t/T)³
        ṡ(t)  = (6t/T² − 6t²/T³)
        s̈(t)  = (6/T² − 12t/T³)

        Returns (s, sdot, sddot) at time t ∈ [0, T].
        """
        t    = np.clip(t, 0, T)
        s    =  3 * (t / T) ** 2 - 2 * (t / T) ** 3
        sd   =  6 * t / T ** 2   - 6 * t ** 2 / T ** 3
        sdd  =  6 / T ** 2       - 12 * t / T ** 3
        return s, sd, sdd

    @staticmethod
    def quintic_time_scaling(T, t):
        """
        Ch.9 §9.2.2.1 – Fifth-order polynomial time scaling.
        Enforces s̈(0) = s̈(T) = 0 (zero jerk at endpoints).

        s(t) = 10(t/T)³ − 15(t/T)⁴ + 6(t/T)⁵
        """
        t   = np.clip(t, 0, T)
        r   = t / T
        s   =  10 * r**3 - 15 * r**4 +  6 * r**5
        sd  = (30 * r**2 - 60 * r**3 + 30 * r**4) / T
        sdd = (60 * r    - 180* r**2 + 120* r**3) / T**2
        return s, sd, sdd

    @staticmethod
    def trapezoidal_time_scaling(T, t, v=None, a=None):
        """
        Ch.9 §9.2.2.2 – Trapezoidal (bang-coast-bang) time scaling.

        If v and a are not supplied they are chosen for minimum-time
        critically-damped profile (ζ = 1):  v = 2/T,  a = 4/T².

        Returns (s, sdot, sddot).
        """
        t = np.clip(t, 0, T)
        if v is None: v = 1.5 / T       # cruise speed
        if a is None: a = v**2 / (v*T - 1 + 1e-9)  # derived from s(T)=1
        a = max(a, 1e-6)

        ta = v / a                       # acceleration phase duration

        if t <= ta:                                    # Acceleration
            return 0.5*a*t**2, a*t, a
        elif t <= T - ta:                              # Coast
            s = v*t - 0.5*v**2/a
            return s, v, 0.0
        else:                                          # Deceleration
            dt  = T - t
            s   = (2*a*v*T - 2*v**2 - a*(t - T)**2) / (2*a)
            sd  = a * (T - t)
            return s, sd, -a

    # ── Joint-space trajectories ──────────────────────────────────────────────

    @staticmethod
    def joint_trajectory_cubic(theta_start, theta_end, T, dt):
        """
        Ch.9 §9.2 – Cubic-polynomial joint-space point-to-point trajectory.

        Returns arrays (times, thetas, dthetas, ddthetas).
        theta_start, theta_end : (n,) start/end joint positions.
        T  : total motion time (s).
        dt : time step (s).
        """
        times   = np.arange(0, T + dt, dt)
        n       = len(theta_start)
        thetas   = np.zeros((len(times), n))
        dthetas  = np.zeros((len(times), n))
        ddthetas = np.zeros((len(times), n))

        delta = np.array(theta_end) - np.array(theta_start)

        for k, t in enumerate(times):
            s, sd, sdd = BronchoKinematics.cubic_time_scaling(T, t)
            thetas[k]   = theta_start + s   * delta
            dthetas[k]  =              sd   * delta
            ddthetas[k] =              sdd  * delta

        return times, thetas, dthetas, ddthetas

    @staticmethod
    def joint_trajectory_quintic(theta_start, theta_end, T, dt):
        """
        Ch.9 §9.2 – Quintic-polynomial joint trajectory (zero endpoint jerk).
        """
        times    = np.arange(0, T + dt, dt)
        n        = len(theta_start)
        thetas   = np.zeros((len(times), n))
        dthetas  = np.zeros((len(times), n))
        ddthetas = np.zeros((len(times), n))
        delta    = np.array(theta_end) - np.array(theta_start)

        for k, t in enumerate(times):
            s, sd, sdd = BronchoKinematics.quintic_time_scaling(T, t)
            thetas[k]   = theta_start + s   * delta
            dthetas[k]  =              sd   * delta
            ddthetas[k] =              sdd  * delta

        return times, thetas, dthetas, ddthetas

    # ── Task-space (SE(3)) trajectories ──────────────────────────────────────

    def screw_trajectory(self, X_start, X_end, T, dt,
                         scaling='cubic'):
        """
        Ch.9 §9.2.1 – Straight-line screw trajectory in SE(3).

        X(s) = X_start · exp( log(X_start⁻¹ · X_end) · s )

        The rotation and translation are coupled (constant screw axis).

        Parameters
        ----------
        X_start, X_end : 4×4 homogeneous transforms
        T   : total time (s)
        dt  : time step (s)
        scaling : 'cubic' | 'quintic' | 'trapezoidal'

        Returns
        -------
        times  : (N,) time array
        X_traj : (N, 4, 4) end-effector configurations
        V_traj : (N, 6)    body twists
        """
        scale_fn = {
            'cubic':       self.cubic_time_scaling,
            'quintic':     self.quintic_time_scaling,
            'trapezoidal': self.trapezoidal_time_scaling,
        }.get(scaling, self.cubic_time_scaling)

        times  = np.arange(0, T + dt, dt)
        X_traj = []
        V_traj = []

        log_X  = self.matrix_log_6(self.trans_inv(X_start) @ X_end)

        for t in times:
            s, sd, _ = scale_fn(T, t)

            # Reconstruct the se(3) matrix from the twist vector
            omega = log_X[:3] * s
            v     = log_X[3:] * s
            S_mat = np.zeros((4, 4))
            S_mat[:3, :3] = self.skew_symmetric(omega)
            S_mat[:3,  3] = v

            X = X_start @ self.matrix_exp_6(log_X, s)
            V = log_X * sd                   # body twist

            X_traj.append(X)
            V_traj.append(V)

        return times, np.array(X_traj), np.array(V_traj)

    def cartesian_trajectory(self, X_start, X_end, T, dt,
                             scaling='cubic'):
        """
        Ch.9 §9.2.1 – Decoupled Cartesian trajectory.
        Translation follows a straight line; rotation follows constant
        angular-velocity screw (separate from translation).

        Returns (times, X_traj, V_traj) same format as screw_trajectory.
        """
        scale_fn = {
            'cubic':       self.cubic_time_scaling,
            'quintic':     self.quintic_time_scaling,
            'trapezoidal': self.trapezoidal_time_scaling,
        }.get(scaling, self.cubic_time_scaling)

        times  = np.arange(0, T + dt, dt)
        R_start, p_start = X_start[:3, :3], X_start[:3, 3]
        R_end,   p_end   = X_end[:3, :3],   X_end[:3, 3]

        log_R = self.matrix_log_3(R_start.T @ R_end)   # rotation log

        X_traj = []
        V_traj = []

        for t in times:
            s, sd, _ = scale_fn(T, t)

            # Translation: straight line
            p = p_start + s * (p_end - p_start)
            # Rotation: constant angular velocity about fixed body axis
            # R(s) = R_start · exp([log_R] · s)
            omega_s    = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]]) * s
            norm_omega = np.linalg.norm(omega_s)
            if norm_omega < 1e-8:
                R = R_start
            else:
                ow     = self.skew_symmetric(omega_s / norm_omega)
                R      = (R_start
                          @ (np.eye(3)
                             + np.sin(norm_omega) * ow
                             + (1 - np.cos(norm_omega)) * ow @ ow))

            X = np.eye(4)
            X[:3, :3] = R
            X[:3,  3] = p

            # Body twist (approximate)
            omega_b = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]]) * sd
            v_b     = (p_end - p_start) * sd
            V_traj.append(np.concatenate((omega_b, v_b)))
            X_traj.append(X)

        return times, np.array(X_traj), np.array(V_traj)

    # ── Via-point trajectories ────────────────────────────────────────────────

    @staticmethod
    def via_point_trajectory(via_positions, via_times, via_velocities=None, dt=0.01):
        """
        Ch.9 §9.3 – Piecewise cubic trajectory through a list of via points.

        Parameters
        ----------
        via_positions  : list of (n,) joint-position arrays, length k
        via_times      : list of times [T1, …, Tk], T1 must be 0
        via_velocities : list of (n,) velocity arrays (optional).
                         If None, zero velocity at start/end, matched
                         at interior via points via natural spline rule.
        dt             : output time step (s)

        Returns
        -------
        times   : (N,) time array
        thetas  : (N, n) positions
        dthetas : (N, n) velocities
        """
        k = len(via_positions)
        n = len(via_positions[0])

        # Default velocities: zero at endpoints, smoothed at interior
        if via_velocities is None:
            via_velocities = [np.zeros(n)] * k
            for i in range(1, k - 1):
                dt_prev = via_times[i]   - via_times[i - 1]
                dt_next = via_times[i + 1] - via_times[i]
                v1 = (np.array(via_positions[i]) - np.array(via_positions[i - 1])) / dt_prev
                v2 = (np.array(via_positions[i + 1]) - np.array(via_positions[i])) / dt_next
                via_velocities[i] = 0.5 * (v1 + v2)

        all_times   = []
        all_thetas  = []
        all_dthetas = []

        for seg in range(k - 1):
            beta_s = np.array(via_positions[seg])
            beta_e = np.array(via_positions[seg + 1])
            vd_s   = np.array(via_velocities[seg])
            vd_e   = np.array(via_velocities[seg + 1])
            T_seg  = via_times[seg + 1] - via_times[seg]
            t_seg  = np.arange(0, T_seg + dt, dt)

            # Cubic polynomial coefficients for each joint
            # a0 + a1*t + a2*t² + a3*t³
            # Boundary: pos(0)=β_s, vel(0)=vd_s, pos(T)=β_e, vel(T)=vd_e
            a0 = beta_s
            a1 = vd_s
            a2 = (3 * (beta_e - beta_s) / T_seg**2
                  - 2 * vd_s / T_seg
                  - vd_e / T_seg)
            a3 = (-2 * (beta_e - beta_s) / T_seg**3
                  + (vd_s + vd_e) / T_seg**2)

            for t in t_seg:
                pos = a0 + a1*t + a2*t**2 + a3*t**3
                vel = a1 + 2*a2*t + 3*a3*t**2
                all_times.append(via_times[seg] + t)
                all_thetas.append(pos)
                all_dthetas.append(vel)

        return (np.array(all_times),
                np.array(all_thetas),
                np.array(all_dthetas))

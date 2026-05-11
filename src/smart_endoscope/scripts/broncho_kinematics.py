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

    def __init__(self, L1=0.1, L2=0.1, L3=0.1, L4=0.1, L5=0.05):
        """
        Initialise the 7-DOF kinematic chain.
        Lengths correspond to the updated URDF segments.
        Total length = 0.45m
        """
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.L5 = L5

        # Home configuration M (end-effector when θ = 0)
        # Tip is located at x = 0.45m
        total_length = L1 + L2 + L3 + L4 + L5
        self.M = np.array([
            [1, 0, 0, total_length],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # ── Screw Axes in the base/space frame ─────────────────────────────
        # S = [ω_x, ω_y, ω_z, v_x, v_y, v_z]
        # v = -ω x q (where q is a point on the axis)

        # 1. Prismatic insertion along X
        self.S1 = np.array([0, 0, 0,  1, 0, 0], dtype=float)

        # 2. Proximal Yaw (Z-axis at x=0)
        self.S2 = np.array([0, 0, 1,  0, 0, 0], dtype=float)

        # 3. Proximal Pitch (Y-axis at x=L1)
        # q = [0.1, 0, 0], ω = [0, 1, 0] -> v = [0, 0, 0.1]
        self.S3 = np.array([0, 1, 0,  0, 0, L1], dtype=float)

        # 4. Mid Yaw (Z-axis at x=L1+L2)
        # q = [0.2, 0, 0], ω = [0, 0, 1] -> v = [0, -0.2, 0]
        self.S4 = np.array([0, 0, 1,  0, -(L1 + L2), 0], dtype=float)

        # 5. Mid Pitch (Y-axis at x=L1+L2+L3)
        # q = [0.3, 0, 0], ω = [0, 1, 0] -> v = [0, 0, 0.3]
        self.S5 = np.array([0, 1, 0,  0, 0, (L1 + L2 + L3)], dtype=float)

        # 6. Distal Yaw (Z-axis at x=L1+L2+L3+L4)
        # q = [0.4, 0, 0], ω = [0, 0, 1] -> v = [0, -0.4, 0]
        self.S6 = np.array([0, 0, 1,  0, -(L1 + L2 + L3 + L4), 0], dtype=float)

        # 7. Distal Roll (X-axis at x=L1+L2+L3+L4)
        # q = [0.4, 0, 0], ω = [1, 0, 0] -> v = [0, 0, 0]
        self.S7 = np.array([1, 0, 0,  0, 0, 0], dtype=float)

        self.S_list = [self.S1, self.S2, self.S3, self.S4, self.S5, self.S6, self.S7]

        # ── Spatial inertia matrices for dynamics (7 links) ─────────────────
        link_masses   = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
        link_inertias = [1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 5e-4, 5e-4] 

        self.Glist = []
        for m, I in zip(link_masses, link_inertias):
            G = np.diag([I, I, I, m, m, m])
            self.Glist.append(G)

        lengths = [0.0, L1, L2, L3, L4, L5, 0.0] 
        self.Mlist = []
        for l in lengths:
            Mi = np.eye(4)
            Mi[0, 3] = l
            self.Mlist.append(Mi)

        self.gravity = np.array([0, 0, -9.81])

    # =========================================================================
    # Rigid-Body Motions
    # =========================================================================

    @staticmethod
    def skew_symmetric(vec3):
        return np.array([
            [0,        -vec3[2],  vec3[1]],
            [vec3[2],   0,       -vec3[0]],
            [-vec3[1],  vec3[0],  0      ],
        ])

    def trans_inv(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3,  3] = -R.T @ p
        return T_inv

    def matrix_exp_6(self, S, theta):
        omega = S[:3]
        v     = S[3:]
        T     = np.eye(4)

        if np.linalg.norm(omega) < 1e-6: 
            T[:3, 3] = v * theta
            return T

        omega_skew = self.skew_symmetric(omega)
        R = (np.eye(3)
             + np.sin(theta) * omega_skew
             + (1 - np.cos(theta)) * omega_skew @ omega_skew)
        G = (np.eye(3) * theta
             + (1 - np.cos(theta)) * omega_skew
             + (theta - np.sin(theta)) * omega_skew @ omega_skew)

        T[:3, :3] = R
        T[:3,  3] = G @ v
        return T

    def adjoint_matrix(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        p_skew = self.skew_symmetric(p)

        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, :3] = p_skew @ R
        AdT[3:, 3:] = R
        return AdT

    def matrix_log_3(self, R):
        acos_input = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(acos_input)

        if np.isclose(theta, 0.0): return np.zeros((3, 3))
        if np.isclose(theta, np.pi):
            if not np.isclose(1 + R[2, 2], 0.0): omega = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif not np.isclose(1 + R[1, 1], 0.0): omega = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else: omega = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            return self.skew_symmetric(omega) * np.pi

        return (theta / (2 * np.sin(theta))) * (R - R.T)

    def matrix_log_6(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        omega_skew = self.matrix_log_3(R)

        if np.allclose(omega_skew, 0): v_theta = p
        else:
            theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
            G_inv = (np.eye(3) - 0.5 * omega_skew + (1.0 / theta - 0.5 / np.tan(theta / 2.0)) * (omega_skew @ omega_skew) / theta)
            v_theta = G_inv @ p

        omega_theta = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        return np.concatenate((omega_theta, v_theta))

    # =========================================================================
    # Forward Kinematics 
    # =========================================================================

    def forward_kinematics_space(self, theta_list):
        T = np.eye(4)
        for S, theta in zip(self.S_list, theta_list):
            T = T @ self.matrix_exp_6(S, theta)
        return T @ self.M

    # =========================================================================
    # Velocity Kinematics & Statics
    # =========================================================================

    def jacobian_space(self, theta_list):
        """
        Dynamically calculates the Space Jacobian for n joints.
        Js(θ) = [S1, Ad_T1(S2), Ad_T12(S3), ... ]
        """
        n = len(theta_list)
        Js = np.zeros((6, n))
        T = np.eye(4)
        
        for i in range(n):
            if i == 0:
                Js[:, i] = self.S_list[i]
            else:
                T = T @ self.matrix_exp_6(self.S_list[i-1], theta_list[i-1])
                Js[:, i] = self.adjoint_matrix(T) @ self.S_list[i]
                
        return Js

    def jacobian_body(self, Js, T_final):
        return self.adjoint_matrix(self.trans_inv(T_final)) @ Js

    def ellipsoid_analysis(self, J):
        J_omega = J[:3, :]
        J_v     = J[3:, :]

        def calculate_mu(A_mat):
            eigs = np.clip(np.linalg.eigvalsh(A_mat), 0.0, None)
            eigs = np.sort(eigs)
            lmin, lmax = eigs[0], eigs[-1]
            mu3 = np.sqrt(max(0.0, np.linalg.det(A_mat)))
            if np.isclose(lmin, 0.0, atol=1e-6): return float('inf'), float('inf'), mu3
            return np.sqrt(lmax / lmin), lmax / lmin, mu3

        mu1w, mu2w, mu3w = calculate_mu(J_omega @ J_omega.T)
        mu1v, mu2v, mu3v = calculate_mu(J_v     @ J_v.T)

        return {
            'angular': {'mu1': mu1w, 'mu2': mu2w, 'mu3': mu3w},
            'linear':  {'mu1': mu1v, 'mu2': mu2v, 'mu3': mu3v},
        }

    # =========================================================================
    # Inverse Kinematics
    # =========================================================================

    def ik_body(self, T_sd, theta_guess, e_omega=0.001, e_v=0.0001, max_iter=50):
        theta = np.array(theta_guess, dtype=float)

        for _ in range(max_iter):
            T_sb = self.forward_kinematics_space(theta)
            T_bd = self.trans_inv(T_sb) @ T_sd
            V_b  = self.matrix_log_6(T_bd)

            if np.linalg.norm(V_b[:3]) < e_omega and np.linalg.norm(V_b[3:]) < e_v:
                return theta, True

            Js   = self.jacobian_space(theta)
            Jb   = self.jacobian_body(Js, T_sb)
            
            # np.linalg.pinv automatically handles the redundancy (7 DOF > 6 spatial dims)
            theta += np.linalg.pinv(Jb) @ V_b

        return theta, False
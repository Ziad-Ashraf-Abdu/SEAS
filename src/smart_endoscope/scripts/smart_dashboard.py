#!/usr/bin/env python3
"""
smart_dashboard.py  –  Smart Bronchoscope ROS 2 Dashboard  (Chapter 11 edition)
=================================================================================

NEW in this version
───────────────────
  AUTO mode  – tick the "AUTO" checkbox and the scope navigates itself,
               advancing forward while bending away from walls, using the
               AutoPilot computed-torque / task-space PI controller.

  Collision Guard (Manual mode) – every keyboard command is filtered by
               CollisionGuard before being sent to Gazebo, so the tip
               never touches a wall even under manual control.

  Surgical dark theme – redesigned UI using a deep-space palette with
               amber accent lines, monospaced telemetry, and a status
               strip that changes colour with robot health.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image as PILImage, ImageTk

from broncho_kinematics import BronchoKinematics
from vision_pipeline import VisionProcessor
from broncho_controller import (
    CollisionGuard, AutoPilot, RobotState,
    estimate_wall_proximity, guard_status_text, auto_status_text,
    GUARD_WALL_THRESHOLD, AUTO_WALL_ZONE,
)


# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG      = "#0a0c0f"
PANEL_BG     = "#0f1318"
BORDER_COL   = "#1e2530"
AMBER        = "#e8a020"
AMBER_BRIGHT = "#f5c040"
CYAN_DIM     = "#2dd4bf"
RED_ALERT    = "#ef4444"
GREEN_OK     = "#22c55e"
YELLOW_WARN  = "#eab308"
TEXT_DIM     = "#4a5568"
TEXT_MUTED   = "#718096"
TEXT_MAIN    = "#e2e8f0"
TEXT_BRIGHT  = "#f8fafc"

STATUS_BAR_H = 28      # px
PANEL_W      = 320     # px


# =============================================================================
# ROS 2 Node
# =============================================================================

class SmartDashboardNode(Node):
    def __init__(self):
        super().__init__('smart_dashboard_node')
        self.cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10,
        )
        self.image_subscriber = self.create_subscription(
            Image,
            '/world/airway_world/model/bronchoscope/link/distal_tip/sensor/cmos_camera/image',
            self._image_cb,
            10,
        )
        self.bridge       = CvBridge()
        self.latest_frame = None

    def _image_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge: {e}")

    def publish_joints(self, state: RobotState):
        msg      = Float64MultiArray()
        msg.data = state.as_list
        self.cmd_publisher.publish(msg)


# =============================================================================
# GUI
# =============================================================================

class SmartDashboardGUI:

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk, ros_node: SmartDashboardNode):
        self.root = root
        self.node = ros_node
        self.root.title("Smart Bronchoscope – ROS 2 Dashboard")
        self.root.configure(bg=DARK_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Engines ──────────────────────────────────────────────────────────
        self.kin     = BronchoKinematics(L1=0.20, L2=0.10, L3=0.05)
        self.vision  = VisionProcessor()
        self.guard   = CollisionGuard(self.kin)
        self.pilot   = AutoPilot(self.kin, goal_x=1.40)

        # ── Robot state ──────────────────────────────────────────────────────
        self.state        = RobotState()
        self.tip_position = np.zeros(3)
        self.ellipsoids   = None
        self.guard_dist   = 1.0
        self.guard_normal = np.array([0.0, 1.0])
        self.guard_margin = 1.0

        # ── Tkinter vars ─────────────────────────────────────────────────────
        self.auto_enabled = tk.BooleanVar(value=False)
        self.ai_enabled   = tk.BooleanVar(value=False)
        self.light_val    = tk.DoubleVar(value=1.0)

        self._build_ui()
        self.root.bind('<KeyPress>', self._on_key)
        self._update_loop()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Root grid ────────────────────────────────────────────────────────
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=PANEL_W, weight=0)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, minsize=STATUS_BAR_H, weight=0)

        # ── Camera canvas ────────────────────────────────────────────────────
        cam_frame = tk.Frame(self.root, bg=DARK_BG, bd=0)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))
        self.canvas = tk.Canvas(
            cam_frame, width=640, height=480,
            bg="#000000", highlightthickness=1,
            highlightbackground=BORDER_COL,
        )
        self.canvas.pack(fill="both", expand=True)

        # ── Right control panel ───────────────────────────────────────────────
        panel = tk.Frame(
            self.root, bg=PANEL_BG, bd=0, width=PANEL_W,
        )
        panel.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=(8, 4))
        panel.grid_propagate(False)
        self._build_panel(panel)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_bar = tk.Label(
            self.root,
            text=" SYSTEM READY",
            bg=PANEL_BG, fg=CYAN_DIM,
            font=("Courier", 10),
            anchor="w", padx=12,
        )
        self.status_bar.grid(row=1, column=0, columnspan=2,
                             sticky="ew", ipady=4)

    def _build_panel(self, parent):
        pad = dict(padx=16)  # only padx here; pady is always passed explicitly

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(
            parent, text="BRONCHOSCOPE", bg=PANEL_BG, fg=AMBER,
            font=("Courier", 15, "bold"),
        ).pack(pady=(18, 0))
        tk.Label(
            parent, text="ROS 2  /  CHAPTER 11 CONTROL", bg=PANEL_BG,
            fg=TEXT_DIM, font=("Courier", 8),
        ).pack(pady=(0, 14))

        self._separator(parent)

        # ── Mode toggles ──────────────────────────────────────────────────────
        mode_frame = tk.Frame(parent, bg=PANEL_BG)
        mode_frame.pack(fill="x", **pad, pady=(10, 6))

        self._toggle(mode_frame, "⚙  AUTO NAVIGATE",
                     self.auto_enabled, AMBER, self._on_auto_toggle,
                     side="left")
        self._toggle(mode_frame, "🧠  AI VISION",
                     self.ai_enabled, CYAN_DIM, None,
                     side="right")

        self._separator(parent)

        # ── Light slider ──────────────────────────────────────────────────────
        tk.Label(parent, text="LIGHT INTENSITY", bg=PANEL_BG,
                 fg=TEXT_MUTED, font=("Courier", 8)).pack(**pad, pady=(10, 2))

        slider_frame = tk.Frame(parent, bg=PANEL_BG)
        slider_frame.pack(fill="x", **pad, pady=(0, 6))
        self.light_slider = tk.Scale(
            slider_frame, variable=self.light_val,
            from_=0.1, to=2.5, resolution=0.05, orient="horizontal",
            bg=PANEL_BG, fg=AMBER, troughcolor=BORDER_COL,
            highlightthickness=0, bd=0, showvalue=False,
            activebackground=AMBER_BRIGHT,
        )
        self.light_slider.pack(fill="x")

        self._separator(parent)

        # ── Controls legend ───────────────────────────────────────────────────
        tk.Label(parent, text="KEYBOARD", bg=PANEL_BG,
                 fg=TEXT_MUTED, font=("Courier", 8)).pack(**pad, pady=(10, 4))
        legend = [
            ("W / S", "Insert  /  Retract"),
            ("A / D", "Curl Left  /  Right"),
            ("T",     "Run IK to Target"),
            ("AUTO",  "Autonomous forward drive"),
        ]
        for key, desc in legend:
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill="x", padx=16, pady=1)
            tk.Label(row, text=key, bg=PANEL_BG, fg=AMBER,
                     font=("Courier", 9, "bold"), width=6, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=PANEL_BG, fg=TEXT_MUTED,
                     font=("Courier", 9), anchor="w").pack(side="left")

        self._separator(parent)

        # ── Telemetry readout ─────────────────────────────────────────────────
        tk.Label(parent, text="TIP POSITION  (m)", bg=PANEL_BG,
                 fg=TEXT_MUTED, font=("Courier", 8)).pack(**pad, pady=(10, 2))
        self.pos_label = tk.Label(
            parent, text="X: 0.000   Y: 0.000   Z: 0.000",
            bg=PANEL_BG, fg=AMBER_BRIGHT, font=("Courier", 12, "bold"),
        )
        self.pos_label.pack(**pad, pady=(0, 8))

        # Joints
        tk.Label(parent, text="JOINT STATE", bg=PANEL_BG,
                 fg=TEXT_MUTED, font=("Courier", 8)).pack(**pad, pady=(4, 2))
        self.joint_label = tk.Label(
            parent, text="INS 0.000  P 0.000  M 0.000  D 0.000",
            bg=PANEL_BG, fg=TEXT_MAIN, font=("Courier", 10),
        )
        self.joint_label.pack(**pad, pady=(0, 6))

        # Manipulability
        tk.Label(parent, text="MANIPULABILITY", bg=PANEL_BG,
                 fg=TEXT_MUTED, font=("Courier", 8)).pack(**pad, pady=(4, 2))
        self.manip_label = tk.Label(
            parent, text="VOL – – –   COND – – –",
            bg=PANEL_BG, fg=TEXT_MAIN, font=("Courier", 10),
        )
        self.manip_label.pack(**pad, pady=(0, 6))

        self._separator(parent)

        # Guard & auto status
        self.guard_label = tk.Label(
            parent, text="GUARD  –",
            bg=PANEL_BG, fg=GREEN_OK, font=("Courier", 9),
            wraplength=280, justify="left",
        )
        self.guard_label.pack(**pad, pady=(8, 4))

        self.auto_label = tk.Label(
            parent, text="AUTO  –",
            bg=PANEL_BG, fg=TEXT_DIM, font=("Courier", 9),
        )
        self.auto_label.pack(**pad, pady=(0, 12))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _separator(self, parent):
        tk.Frame(parent, bg=BORDER_COL, height=1).pack(
            fill="x", padx=10, pady=4,
        )

    def _toggle(self, parent, text, var, color, cmd, side):
        cb = tk.Checkbutton(
            parent, text=text, variable=var,
            bg=PANEL_BG, fg=color, selectcolor=DARK_BG,
            activebackground=PANEL_BG, activeforeground=color,
            font=("Courier", 9, "bold"),
            command=cmd,
        )
        cb.pack(side=side)

    # ── Event callbacks ────────────────────────────────────────────────────────

    def _on_auto_toggle(self):
        if self.auto_enabled.get():
            self.pilot.reset(goal_x=1.40)
            self.guard.reset()
            print("[AUTO] Autopilot engaged – scope will advance autonomously.")
        else:
            print("[AUTO] Autopilot disengaged – returning to manual control.")

    def _on_key(self, event):
        """Handle keyboard input in MANUAL mode only."""
        if self.auto_enabled.get():
            return  # keyboard locked while AUTO is running

        key = event.char.lower()
        s   = self.state

        delta_ins = delta_prox = delta_mid = delta_dis = 0.0

        if key == 'w':
            delta_ins = +0.01
        elif key == 's':
            delta_ins = -0.01
        elif key == 'd':
            delta_dis  = -0.05
            delta_mid  = -0.05 * 0.70
            delta_prox = -0.05 * 0.40
        elif key == 'a':
            delta_dis  = +0.05
            delta_mid  = +0.05 * 0.70
            delta_prox = +0.05 * 0.40
        elif key == 't':
            self._run_ik()
            return
        else:
            return

        # ── Collision Guard filters the command (Chapter 11 §11.3.3) ──────────
        frame = self.node.latest_frame
        new_state = self.guard.filter_command(
            s,
            delta_ins, delta_prox, delta_mid, delta_dis,
            frame=frame,
        )

        self.state = new_state
        self._update_kinematics()
        self.node.publish_joints(self.state)

    def _run_ik(self):
        """Chapter 6 IK to a hard-coded target (keyboard 'T')."""
        T_sd = np.array([
            [ 0.877, -0.479, 0.0, 0.800],
            [ 0.479,  0.877, 0.0, 0.150],
            [ 0.0,    0.0,   1.0, 0.000],
            [ 0.0,    0.0,   0.0, 1.000],
        ])
        new_theta, converged = self.kin.ik_body(
            T_sd, self.state.as_list
        )
        if converged:
            self.state = RobotState(*new_theta).clipped()
            print(f"[IK] Converged  → {new_theta}")
        else:
            print("[IK] Failed to converge – target may be out of reach.")
        self._update_kinematics()
        self.node.publish_joints(self.state)

    # ── Kinematics update ─────────────────────────────────────────────────────

    def _update_kinematics(self):
        th = self.state.as_list
        T1 = self.kin.matrix_exp_6(self.kin.S1, th[0])
        T2 = self.kin.matrix_exp_6(self.kin.S2, th[1])
        T3 = self.kin.matrix_exp_6(self.kin.S3, th[2])
        T4 = self.kin.matrix_exp_6(self.kin.S4, th[3])
        T_final = np.dot(np.dot(np.dot(T1, T2), T3), T4).dot(self.kin.M)

        self.tip_position = T_final[:3, 3]
        Js = self.kin.jacobian_space(th)
        self.ellipsoids   = self.kin.ellipsoid_analysis(Js)

        # Update telemetry labels
        p = self.tip_position
        self.pos_label.config(
            text=f"X: {p[0]:+.3f}   Y: {p[1]:+.3f}   Z: {p[2]:+.3f}"
        )
        s = self.state
        self.joint_label.config(
            text=f"INS {s.insertion:+.3f}  P {s.proximal:+.3f}  "
                 f"M {s.mid:+.3f}  D {s.distal:+.3f}"
        )
        el = self.ellipsoids
        vol  = el['linear']['mu3']
        cond = el['linear']['mu1']
        cond_str = "SING" if cond == float('inf') else f"{cond:.2f}"
        self.manip_label.config(
            text=f"VOL {vol:.4f}   COND {cond_str}",
            fg=RED_ALERT if cond == float('inf') else TEXT_MAIN,
        )

    # ── HUD overlay on camera frame ───────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Crosshair
        cv2.line(frame, (cx - 22, cy), (cx + 22, cy), (0, 210, 80), 1)
        cv2.line(frame, (cx, cy - 22), (cx, cy + 22), (0, 210, 80), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 210, 80), 1)

        # Corner reticle brackets
        blen, boff = 18, 3
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            ox = cx + sx * (w // 4)
            oy = cy + sy * (h // 4)
            cv2.line(frame, (ox, oy), (ox + sx * blen, oy), (50, 180, 255), 1)
            cv2.line(frame, (ox, oy), (ox, oy + sy * blen), (50, 180, 255), 1)

        # Bottom telemetry bar
        s   = self.state
        el  = self.ellipsoids
        if el:
            vol  = el['linear']['mu3']
            cond = el['linear']['mu1']
            cond_str = "SING!" if cond == float('inf') else f"{cond:.2f}"
            singular = (cond == float('inf'))
        else:
            vol, cond_str, singular = 0.0, "N/A", False

        mode_str = "AUTO" if self.auto_enabled.get() else "MANU"
        tele = (
            f"{mode_str} | "
            f"INS {s.insertion:+.3f}m  "
            f"P={s.proximal:+.4f}  M={s.mid:+.4f}  D={s.distal:+.4f}  |  "
            f"VOL {vol:.4f}  COND {cond_str}"
        )

        bar_y = h - STATUS_BAR_H
        cv2.rectangle(frame, (0, bar_y), (w, h), (10, 14, 20), -1)
        text_col = (0, 80, 240) if singular else (20, 220, 140)
        cv2.putText(frame, tele, (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1,
                    cv2.LINE_AA)

        # Guard distance bar (left edge)
        dist = self.guard_dist
        bar_frac = np.clip(dist / 0.5, 0.0, 1.0)
        bar_full  = h - STATUS_BAR_H - 10
        bar_pix   = int(bar_frac * bar_full)
        bar_col   = (0, 220, 80) if dist >= GUARD_WALL_THRESHOLD else (
                    (0, 200, 255) if dist >= GUARD_WALL_THRESHOLD * 0.5
                    else (0, 60, 255))
        cv2.rectangle(frame, (4, h - STATUS_BAR_H - 10 - bar_pix),
                      (10, h - STATUS_BAR_H - 10), bar_col, -1)
        cv2.rectangle(frame, (4, 10), (10, h - STATUS_BAR_H - 10),
                      (30, 40, 50), 1)
        cv2.putText(frame, "GRD", (2, h - STATUS_BAR_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)

        # AUTO goal progress bar (right edge)
        if self.auto_enabled.get():
            progress = np.clip(s.insertion / self.pilot.goal_x, 0.0, 1.0)
            prog_pix = int(progress * bar_full)
            cv2.rectangle(frame,
                          (w - 10, h - STATUS_BAR_H - 10 - prog_pix),
                          (w - 4,  h - STATUS_BAR_H - 10),
                          (20, 160, 255), -1)
            cv2.rectangle(frame, (w - 10, 10), (w - 4, h - STATUS_BAR_H - 10),
                          (30, 40, 50), 1)
            cv2.putText(frame, "FWD", (w - 13, h - STATUS_BAR_H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)

        return frame

    # ── Main update loop ──────────────────────────────────────────────────────

    def _update_loop(self):
        # 1. Spin ROS
        rclpy.spin_once(self.node, timeout_sec=0.01)

        frame = self.node.latest_frame

        # 2. Proximity sensing (used by both guard and auto)
        if frame is not None:
            dist, normal, margin = estimate_wall_proximity(
                frame, self.tip_position[:2]
            )
            self.guard_dist   = dist
            self.guard_normal = normal
            self.guard_margin = margin

        # 3. AUTO PILOT step
        if self.auto_enabled.get() and not self.pilot.is_done:
            self.state = self.pilot.step(self.state, frame=frame)
            self._update_kinematics()
            self.node.publish_joints(self.state)

        # 4. Guard & auto status labels
        self.guard_label.config(
            text=guard_status_text(self.guard_dist, self.guard_margin),
            fg=(RED_ALERT if self.guard_dist < GUARD_WALL_THRESHOLD * 0.5
                else YELLOW_WARN if self.guard_dist < GUARD_WALL_THRESHOLD
                else GREEN_OK),
        )
        auto_txt = (auto_status_text(self.pilot, self.state)
                    if self.auto_enabled.get() else "AUTO  –")
        auto_col = (AMBER if self.auto_enabled.get() else TEXT_DIM)
        self.auto_label.config(text=auto_txt, fg=auto_col)

        # 5. Render frame
        if frame is not None:
            frame = cv2.convertScaleAbs(
                frame, alpha=self.light_val.get(), beta=0
            )
            if self.ai_enabled.get():
                frame = self.vision.process_frame(frame)
            frame = self._draw_hud(frame)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = PILImage.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk  # keep reference

        # 6. Status bar
        mode    = "AUTO" if self.auto_enabled.get() else "MANUAL"
        ai_str  = "AI ON" if self.ai_enabled.get() else "AI OFF"
        sb_col  = AMBER if self.auto_enabled.get() else CYAN_DIM
        self.status_bar.config(
            text=f"  MODE: {mode}  |  {ai_str}  |  "
                 f"GUARD dist={self.guard_dist:.3f}m  |  "
                 f"TIP X={self.tip_position[0]:.3f}m",
            fg=sb_col,
        )

        # Re-schedule: ~30 fps in manual, ~15 fps in auto (heavier computation)
        delay = 66 if self.auto_enabled.get() else 33
        self.root.after(delay, self._update_loop)

    # ── Teardown ──────────────────────────────────────────────────────────────

    def _on_close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


# =============================================================================
# Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)

    ros_node = SmartDashboardNode()
    root     = tk.Tk()
    root.resizable(True, True)

    SmartDashboardGUI(root, ros_node)

    root.focus_set()
    root.mainloop()


if __name__ == '__main__':
    main()

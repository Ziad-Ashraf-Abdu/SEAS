#!/usr/bin/env python3
"""
smart_dashboard.py  –  Smart Bronchoscope ROS 2 Dashboard  (Clinical Edition)
===============================================================================

Redesigned for clinical users (doctors & medical students).
All engineering internals (ROS2, kinematics, autopilot) are unchanged.
Only the presentation layer has been updated for medical usability.

Controls
────────
  W / S   Advance deeper / Withdraw scope
  A / D   Steer left / Steer right
  R / F   Bend upward / Bend downward
  Q / E   Rotate tip clockwise / counter-clockwise
  T       Move to preset target position (IK)
  AUTO    Let the system navigate automatically
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
    GUARD_WALL_THRESHOLD, AUTO_WALL_ZONE, VERTICAL_STEP,
)


# ── Clinical Colour Palette ───────────────────────────────────────────────────
# Deep navy-black surgical environment — familiar to anyone who's seen an OR monitor
DARK_BG       = "#06080d"          # near-black background
PANEL_BG      = "#0c1018"          # panel background
CARD_BG       = "#111722"          # card / section background
BORDER_COL    = "#1c2535"          # subtle borders
BORDER_LIGHT  = "#253040"

# Clinical traffic-light system (universal to medical staff)
SAFE_GREEN    = "#1eba6a"          # safe / all-clear
CAUTION_AMBER = "#f5a623"          # caution / attention needed
DANGER_RED    = "#e84040"          # warning / stop
INFO_BLUE     = "#4db8e8"          # informational cyan-blue

# Accent & highlight
SCOPE_TEAL    = "#00c4b0"          # scope / position accent
GOLD          = "#d4a843"          # headings, active states
GOLD_BRIGHT   = "#f0c060"

# Text hierarchy
TEXT_BRIGHT   = "#f0f4f8"          # primary readout
TEXT_MAIN     = "#b8c8d8"          # body text
TEXT_MUTED    = "#607080"          # labels, secondary
TEXT_DIM      = "#384858"          # very dim / disabled

STATUS_BAR_H  = 32
PANEL_W       = 340


# ── Human-readable translations ───────────────────────────────────────────────

def depth_cm(m_val: float) -> str:
    """Convert metres to centimetres for clinical display."""
    return f"{m_val * 100:.1f} cm"


def proximity_label(dist: float) -> tuple[str, str]:
    """Return (label, colour) for wall proximity in plain English."""
    if dist < GUARD_WALL_THRESHOLD * 0.5:
        return "⚠  WALL VERY CLOSE — Scope protected", DANGER_RED
    elif dist < GUARD_WALL_THRESHOLD:
        return "▲  Approaching airway wall — Slowing", CAUTION_AMBER
    else:
        return "✔  Clear — Safe to advance", SAFE_GREEN


def navigation_status(auto_on: bool, pilot, state) -> tuple[str, str]:
    """Return (label, colour) for navigation mode."""
    if not auto_on:
        return "Manual Control  (keyboard active)", TEXT_MUTED
    if pilot.is_done:
        return "✔  Target depth reached", SAFE_GREEN
    pct = int(np.clip(state.insertion / pilot.goal_x, 0.0, 1.0) * 100)
    return f"Auto-navigating … {pct}% to target", SCOPE_TEAL


# =============================================================================
# ROS 2 Node  (unchanged)
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
# GUI  (clinical redesign)
# =============================================================================

class SmartDashboardGUI:

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk, ros_node: SmartDashboardNode):
        self.root = root
        self.node = ros_node
        self.root.title("Smart Bronchoscope  –  Clinical Control System")
        self.root.configure(bg=DARK_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Engines ──────────────────────────────────────────────────────────
        self.kin    = BronchoKinematics(L1=0.20, L2=0.10, L3=0.05)
        self.vision = VisionProcessor()
        self.guard  = CollisionGuard(self.kin)
        self.pilot  = AutoPilot(self.kin, goal_x=1.40)

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
        self.target_x_val = tk.DoubleVar(value=1.0)

        self._build_ui()
        self.root.bind_all('<KeyPress>', self._on_key)
        self.root.focus_force()
        self._update_loop()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=PANEL_W, weight=0)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, minsize=STATUS_BAR_H, weight=0)

        # ── Camera feed area ─────────────────────────────────────────────────
        cam_outer = tk.Frame(self.root, bg=DARK_BG, bd=0)
        cam_outer.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))

        # Label above camera
        cam_header = tk.Frame(cam_outer, bg=DARK_BG)
        cam_header.pack(fill="x", pady=(0, 4))
        tk.Label(
            cam_header, text="ENDOSCOPE CAMERA FEED",
            bg=DARK_BG, fg=TEXT_MUTED,
            font=("Helvetica", 9, "bold"), anchor="w",
        ).pack(side="left")
        self.cam_status_dot = tk.Label(
            cam_header, text="●  LIVE",
            bg=DARK_BG, fg=SAFE_GREEN,
            font=("Helvetica", 9, "bold"),
        )
        self.cam_status_dot.pack(side="right")

        self.canvas = tk.Canvas(
            cam_outer, width=640, height=480,
            bg="#000000", highlightthickness=2,
            highlightbackground=BORDER_LIGHT,
        )
        self.canvas.pack(fill="both", expand=True)

        # ── Right control panel ───────────────────────────────────────────────
        panel = tk.Frame(self.root, bg=PANEL_BG, bd=0, width=PANEL_W)
        panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
        panel.grid_propagate(False)
        self._build_panel(panel)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_bar = tk.Label(
            self.root,
            text="  System initialising …",
            bg="#090d14", fg=INFO_BLUE,
            font=("Helvetica", 10),
            anchor="w", padx=14,
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", ipady=6)

    def _build_panel(self, parent):
        px = dict(padx=14)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(parent, bg="#0a1020", pady=0)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="SMART BRONCHOSCOPE",
            bg="#0a1020", fg=GOLD,
            font=("Helvetica", 14, "bold"),
        ).pack(pady=(16, 0))
        tk.Label(
            hdr, text="Clinical Navigation System",
            bg="#0a1020", fg=TEXT_MUTED,
            font=("Helvetica", 9),
        ).pack(pady=(2, 14))

        tk.Frame(parent, bg=BORDER_COL, height=1).pack(fill="x")

        # ── SAFETY MONITOR card ──────────────────────────────────────────────
        self._section_label(parent, "🛡  SAFETY MONITOR")

        safety_card = tk.Frame(parent, bg=CARD_BG, bd=0)
        safety_card.pack(fill="x", **px, pady=(4, 8))

        self.guard_label = tk.Label(
            safety_card,
            text="✔  Clear — Safe to advance",
            bg=CARD_BG, fg=SAFE_GREEN,
            font=("Helvetica", 10, "bold"),
            anchor="w", wraplength=290, justify="left", padx=10, pady=8,
        )
        self.guard_label.pack(fill="x")

        tk.Frame(parent, bg=BORDER_COL, height=1).pack(fill="x", **px)

        # ── SCOPE POSITION card ──────────────────────────────────────────────
        self._section_label(parent, "📍  SCOPE POSITION")

        pos_card = tk.Frame(parent, bg=CARD_BG, bd=0)
        pos_card.pack(fill="x", **px, pady=(4, 8))

        # Insertion depth — the most clinically relevant value, shown large
        tk.Label(
            pos_card, text="Insertion Depth",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 8), anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 0))

        self.depth_label = tk.Label(
            pos_card,
            text="0.0 cm",
            bg=CARD_BG, fg=GOLD_BRIGHT,
            font=("Helvetica", 24, "bold"),
            anchor="w",
        )
        self.depth_label.pack(fill="x", padx=10, pady=(0, 4))

        # Tip coordinates in smaller text
        tk.Label(
            pos_card, text="Tip co-ordinates (X / Y / Z)",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 8), anchor="w",
        ).pack(fill="x", padx=10)

        self.pos_label = tk.Label(
            pos_card,
            text="0.000  /  0.000  /  0.000  m",
            bg=CARD_BG, fg=SCOPE_TEAL,
            font=("Courier", 11),
            anchor="w",
        )
        self.pos_label.pack(fill="x", padx=10, pady=(2, 4))

        # Scope flexibility (manipulability) — shown as plain health indicator
        tk.Label(
            pos_card, text="Scope manoeuvrability",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 8), anchor="w",
        ).pack(fill="x", padx=10)

        self.manip_label = tk.Label(
            pos_card,
            text="Good",
            bg=CARD_BG, fg=SAFE_GREEN,
            font=("Helvetica", 10, "bold"),
            anchor="w",
        )
        self.manip_label.pack(fill="x", padx=10, pady=(2, 8))

        tk.Frame(parent, bg=BORDER_COL, height=1).pack(fill="x", **px)

        # ── NAVIGATION MODE card ─────────────────────────────────────────────
        self._section_label(parent, "🧭  NAVIGATION MODE")

        mode_card = tk.Frame(parent, bg=CARD_BG, bd=0)
        mode_card.pack(fill="x", **px, pady=(4, 8))

        # Mode toggles — bigger, clearer buttons
        toggle_row = tk.Frame(mode_card, bg=CARD_BG)
        toggle_row.pack(fill="x", padx=8, pady=(8, 4))

        self.auto_btn = tk.Checkbutton(
            toggle_row,
            text="  Auto-Navigate",
            variable=self.auto_enabled,
            bg=CARD_BG, fg=GOLD,
            selectcolor=DARK_BG,
            activebackground=CARD_BG, activeforeground=GOLD_BRIGHT,
            font=("Helvetica", 10, "bold"),
            command=self._on_auto_toggle,
        )
        self.auto_btn.pack(side="left")

        self.ai_btn = tk.Checkbutton(
            toggle_row,
            text="  AI Vision",
            variable=self.ai_enabled,
            bg=CARD_BG, fg=SCOPE_TEAL,
            selectcolor=DARK_BG,
            activebackground=CARD_BG, activeforeground=INFO_BLUE,
            font=("Helvetica", 10, "bold"),
        )
        self.ai_btn.pack(side="right")

        # Navigation status text
        self.auto_label = tk.Label(
            mode_card,
            text="Manual Control  (keyboard active)",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 9), anchor="w", padx=10,
        )
        self.auto_label.pack(fill="x", pady=(2, 4))

        # Target depth slider — labelled for doctors
        tk.Label(
            mode_card, text="Auto-navigate target depth",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 8), anchor="w",
        ).pack(fill="x", padx=10, pady=(4, 0))

        depth_row = tk.Frame(mode_card, bg=CARD_BG)
        depth_row.pack(fill="x", padx=10, pady=(0, 4))

        tk.Label(depth_row, text="10 cm", bg=CARD_BG,
                 fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="left")

        self.target_slider = tk.Scale(
            depth_row, variable=self.target_x_val,
            from_=0.1, to=1.5, resolution=0.01, orient="horizontal",
            bg=CARD_BG, fg=SCOPE_TEAL, troughcolor=BORDER_COL,
            highlightthickness=0, bd=0, showvalue=False,
            activebackground=SCOPE_TEAL,
        )
        self.target_slider.pack(side="left", fill="x", expand=True)

        tk.Label(depth_row, text="150 cm", bg=CARD_BG,
                 fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="right")

        # Show selected target depth live
        self.target_depth_label = tk.Label(
            mode_card, text="Target: 100.0 cm",
            bg=CARD_BG, fg=GOLD,
            font=("Helvetica", 9, "bold"), anchor="center",
        )
        self.target_depth_label.pack(pady=(0, 8))
        self.target_x_val.trace_add("write", self._update_target_label)

        tk.Frame(parent, bg=BORDER_COL, height=1).pack(fill="x", **px)

        # ── ILLUMINATION card ────────────────────────────────────────────────
        self._section_label(parent, "💡  ILLUMINATION")

        light_card = tk.Frame(parent, bg=CARD_BG, bd=0)
        light_card.pack(fill="x", **px, pady=(4, 8))

        light_row = tk.Frame(light_card, bg=CARD_BG)
        light_row.pack(fill="x", padx=10, pady=(8, 8))

        tk.Label(light_row, text="Dim", bg=CARD_BG,
                 fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="left")

        self.light_slider = tk.Scale(
            light_row, variable=self.light_val,
            from_=0.1, to=2.5, resolution=0.05, orient="horizontal",
            bg=CARD_BG, fg=GOLD, troughcolor=BORDER_COL,
            highlightthickness=0, bd=0, showvalue=False,
            activebackground=GOLD_BRIGHT,
        )
        self.light_slider.pack(side="left", fill="x", expand=True)

        tk.Label(light_row, text="Bright", bg=CARD_BG,
                 fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="right")

        tk.Frame(parent, bg=BORDER_COL, height=1).pack(fill="x", **px)

        # ── KEYBOARD CONTROLS legend ──────────────────────────────────────────
        self._section_label(parent, "⌨  KEYBOARD CONTROLS")

        keys_card = tk.Frame(parent, bg=CARD_BG, bd=0)
        keys_card.pack(fill="x", **px, pady=(4, 12))

        controls = [
            ("W",     "Advance scope deeper"),
            ("S",     "Withdraw scope"),
            ("A / D", "Steer left / right"),
            ("R / F", "Bend upward / downward"),
            ("Q / E", "Rotate tip (twist)"),
            ("Z / X", "Shift scope UP / DOWN"),
            ("T",     "Jump to preset position"),
        ]
        for key, desc in controls:
            row = tk.Frame(keys_card, bg=CARD_BG)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(
                row, text=key,
                bg="#182030", fg=GOLD_BRIGHT,
                font=("Courier", 9, "bold"),
                width=7, anchor="center",
                relief="flat", padx=4, pady=2,
            ).pack(side="left")
            tk.Label(
                row, text=desc,
                bg=CARD_BG, fg=TEXT_MAIN,
                font=("Helvetica", 9), anchor="w",
            ).pack(side="left", padx=(8, 0))

        # small bottom padding
        tk.Frame(parent, bg=PANEL_BG, height=4).pack(fill="x")

    # ── Small helpers ─────────────────────────────────────────────────────────

    def _section_label(self, parent, text: str):
        tk.Label(
            parent, text=text,
            bg=PANEL_BG, fg=TEXT_MUTED,
            font=("Helvetica", 8, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))

    def _update_target_label(self, *_):
        cm = self.target_x_val.get() * 100
        self.target_depth_label.config(text=f"Target: {cm:.0f} cm")

    # ── Event callbacks ────────────────────────────────────────────────────────

    def _on_auto_toggle(self):
        if self.auto_enabled.get():
            user_target = self.target_x_val.get()
            self.pilot.reset(goal_x=user_target)
            self.guard.reset()
            print(f"[AUTO] Autopilot engaged – advancing to {user_target:.2f}m.")
        else:
            print("[AUTO] Autopilot disengaged – returning to manual control.")

    def _on_key(self, event):
        print(f"[DEBUG] Key pressed: '{event.char}'")

        if self.auto_enabled.get():
            print("[DEBUG] In AUTO mode — keyboard ignored.")
            return

        key = event.char.lower()
        d_theta = [0.0] * 8   # 8 DOF: vertical + 7 arm joints

        if   key == 'w': d_theta[1] = +0.005    # insertion (index 1)
        elif key == 's': d_theta[1] = -0.005
        elif key == 'a':
            d_theta[2], d_theta[4], d_theta[6] = 0.01, 0.015, 0.025   # yaw joints
        elif key == 'd':
            d_theta[2], d_theta[4], d_theta[6] = -0.01, -0.015, -0.025
        elif key == 'r':
            d_theta[3], d_theta[5] = 0.015, 0.025    # pitch joints
        elif key == 'f':
            d_theta[3], d_theta[5] = -0.015, -0.025
        elif key == 'q': d_theta[7] = +0.03      # distal roll (index 7)
        elif key == 'e': d_theta[7] = -0.03
        elif key == 'z':
            # Raise entire arm in world Z — only vertical_joint (index 0)
            d_theta[0] = +VERTICAL_STEP
        elif key == 'x':
            # Lower entire arm in world Z — only vertical_joint (index 0)
            d_theta[0] = -VERTICAL_STEP
        elif key == 't':
            self._run_ik()
            return
        else:
            return

        frame = self.node.latest_frame
        self.state = self.guard.filter_command(self.state, d_theta, frame=frame)
        self._update_kinematics()
        self.node.publish_joints(self.state)

    def _run_ik(self):
        """IK to a hard-coded target (keyboard 'T'). Vertical joint is preserved."""
        T_sd = np.array([
            [ 0.877, -0.479, 0.0, 0.800],
            [ 0.479,  0.877, 0.0, 0.150],
            [ 0.0,    0.0,   1.0, 0.000],
            [ 0.0,    0.0,   0.0, 1.000],
        ])
        arm_guess = self.state.as_list[1:]   # 7 arm joints only
        new_arm, converged = self.kin.ik_body(T_sd, arm_guess)
        if converged:
            # Keep the current vertical position; only update arm joints
            self.state = RobotState(self.state.vertical, *new_arm).clipped()
            print(f"[IK] Converged → {new_arm}")
        else:
            print("[IK] Failed to converge.")
        self._update_kinematics()
        self.node.publish_joints(self.state)

    # ── Kinematics update ─────────────────────────────────────────────────────

    def _update_kinematics(self):
        # vertical_joint is index 0 — slice it off; BronchoKinematics expects 7 arm joints
        arm_thetas = self.state.as_list[1:]

        T1 = self.kin.matrix_exp_6(self.kin.S1, arm_thetas[0])
        T2 = self.kin.matrix_exp_6(self.kin.S2, arm_thetas[1])
        T3 = self.kin.matrix_exp_6(self.kin.S3, arm_thetas[2])
        T4 = self.kin.matrix_exp_6(self.kin.S4, arm_thetas[3])
        T_final = np.dot(np.dot(np.dot(T1, T2), T3), T4).dot(self.kin.M)

        self.tip_position = T_final[:3, 3]
        Js = self.kin.jacobian_space(arm_thetas)
        self.ellipsoids = self.kin.ellipsoid_analysis(Js)

        # Insertion depth in centimetres
        self.depth_label.config(
            text=depth_cm(self.state.insertion),
        )

        # Tip X/Y/Z
        p = self.tip_position
        self.pos_label.config(
            text=f"{p[0]:+.3f}  /  {p[1]:+.3f}  /  {p[2]:+.3f}  m"
        )

        # Manoeuvrability in plain language
        el   = self.ellipsoids
        cond = el['linear']['mu1']
        if cond == float('inf'):
            manip_text, manip_col = "Limited — near boundary", CAUTION_AMBER
        elif cond > 50:
            manip_text, manip_col = "Reduced", CAUTION_AMBER
        else:
            manip_text, manip_col = "Good", SAFE_GREEN
        self.manip_label.config(text=manip_text, fg=manip_col)
    # ── HUD overlay on camera frame ───────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # ── Crosshair (surgical green) ────────────────────────────────────────
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 200, 80), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 200, 80), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 80), 1)

        # ── Corner orientation brackets ───────────────────────────────────────
        blen = 16
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            ox = cx + sx * (w // 4)
            oy = cy + sy * (h // 4)
            cv2.line(frame, (ox, oy), (ox + sx * blen, oy), (60, 180, 255), 1)
            cv2.line(frame, (ox, oy), (ox, oy + sy * blen), (60, 180, 255), 1)

        # ── Bottom status bar ────────────────────────────────────────────────
        s  = self.state
        el = self.ellipsoids

        dist = self.guard_dist
        if dist < GUARD_WALL_THRESHOLD * 0.5:
            safety_str = "⚠ WALL NEAR"
            bar_col_cv = (0, 60, 240)
        elif dist < GUARD_WALL_THRESHOLD:
            safety_str = "▲ CAUTION"
            bar_col_cv = (0, 180, 220)
        else:
            safety_str = "✔ CLEAR"
            bar_col_cv = (0, 200, 80)

        mode_str = "AUTO" if self.auto_enabled.get() else "MANUAL"

        tele = (
            f"{mode_str}  |  "
            f"Depth {depth_cm(s.insertion)}  |  "
            f"Safety: {safety_str}"
        )

        bar_y = h - STATUS_BAR_H
        cv2.rectangle(frame, (0, bar_y), (w, h), (8, 12, 20), -1)
        cv2.putText(frame, tele, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 215, 230), 1,
                    cv2.LINE_AA)

        # ── Left edge: wall clearance bar ────────────────────────────────────
        bar_full = h - STATUS_BAR_H - 14
        bar_frac = np.clip(dist / 0.5, 0.0, 1.0)
        bar_pix  = int(bar_frac * bar_full)
        cv2.rectangle(frame, (4, h - STATUS_BAR_H - 10 - bar_pix),
                      (10, h - STATUS_BAR_H - 10), bar_col_cv, -1)
        cv2.rectangle(frame, (4, 10), (10, h - STATUS_BAR_H - 10),
                      (30, 42, 55), 1)
        cv2.putText(frame, "SAFE", (0, h - STATUS_BAR_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 90, 100), 1)

        # ── Right edge: progress bar (AUTO only) ──────────────────────────────
        if self.auto_enabled.get():
            progress = np.clip(s.insertion / self.pilot.goal_x, 0.0, 1.0)
            prog_pix = int(progress * bar_full)
            cv2.rectangle(frame,
                          (w - 10, h - STATUS_BAR_H - 10 - prog_pix),
                          (w - 4,  h - STATUS_BAR_H - 10),
                          (20, 160, 255), -1)
            cv2.rectangle(frame, (w - 10, 10), (w - 4, h - STATUS_BAR_H - 10),
                          (30, 42, 55), 1)
            cv2.putText(frame, "PROG", (w - 15, h - STATUS_BAR_H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 90, 100), 1)

        return frame

    # ── Main update loop ──────────────────────────────────────────────────────

    def _update_loop(self):
        # 1. Spin ROS
        rclpy.spin_once(self.node, timeout_sec=0.01)

        frame = self.node.latest_frame

        # 2. Proximity sensing
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

        # 4. Safety & navigation labels (plain English)
        guard_txt, guard_col = proximity_label(self.guard_dist)
        self.guard_label.config(text=guard_txt, fg=guard_col)

        nav_txt, nav_col = navigation_status(
            self.auto_enabled.get(), self.pilot, self.state
        )
        self.auto_label.config(text=nav_txt, fg=nav_col)

        # 5. Render camera frame
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
            self.canvas.imgtk = imgtk

        # 6. Bottom status bar
        mode   = "AUTO-NAVIGATE" if self.auto_enabled.get() else "MANUAL CONTROL"
        ai_str = "AI Vision ON" if self.ai_enabled.get() else "AI Vision OFF"
        depth  = depth_cm(self.tip_position[0])
        vert   = f"{self.state.vertical * 100:+.1f} cm"
        sb_col = GOLD if self.auto_enabled.get() else INFO_BLUE
        _, safety_col = proximity_label(self.guard_dist)
        self.status_bar.config(
            text=f"  {mode}  ·  {ai_str}  ·  Depth: {depth}  ·  "
                 f"Arm height: {vert}  ·  Wall clearance: {self.guard_dist * 100:.1f} cm",
            fg=sb_col,
        )

        # 7. Camera live indicator
        dot_col = SAFE_GREEN if frame is not None else DANGER_RED
        dot_txt = "●  LIVE" if frame is not None else "●  NO SIGNAL"
        self.cam_status_dot.config(fg=dot_col, text=dot_txt)

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
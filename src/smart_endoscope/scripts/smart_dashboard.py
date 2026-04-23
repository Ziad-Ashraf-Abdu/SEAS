#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import threading
from datetime import datetime

class SmartEndoscopeUI(Node):
    def __init__(self):
        super().__init__('smart_endoscope_ui')
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.img_sub = self.create_subscription(Image, '/world/airway_world/model/bronchoscope/link/distal_tip/sensor/cmos_camera/image', self.image_callback, 10)
        
        self.insertion_pos = 0.0
        self.proximal_pos = 0.0
        self.mid_pos = 0.0
        self.distal_pos = 0.0
        
        self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.bg_color = "#1E1E2E"
        self.panel_color = "#282A36"
        self.accent_color = "#00B4D8"
        self.text_color = "#F8F8F2"
        
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.analyze_mode = tk.BooleanVar(value=False)
        
        self.root.title("Akatsuki Endoscopy | Clinical Interface")
        self.root.geometry("1000x550")
        self.root.configure(bg=self.bg_color)
        self.root.resizable(False, False)
        
        vid_frame = tk.Frame(self.root, bg=self.bg_color, bd=0)
        vid_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.video_label = tk.Label(vid_frame, bg="black", bd=2, relief="solid", highlightbackground=self.accent_color)
        self.video_label.pack()
        
        control_frame = tk.Frame(self.root, bg=self.panel_color, bd=0, highlightthickness=1, highlightbackground="#444")
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(control_frame, text="S Y S T E M   S T A T U S", font=("Segoe UI", 14, "bold"), fg=self.accent_color, bg=self.panel_color).pack(pady=(20, 5))
        tk.Frame(control_frame, bg="#444", height=1).pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(control_frame, text="ILLUMINATION INTENSITY", font=("Segoe UI", 9, "bold"), fg="#8BE9FD", bg=self.panel_color).pack(pady=(15, 0))
        self.light_slider = tk.Scale(control_frame, from_=0.1, to=2.5, resolution=0.1, orient=tk.HORIZONTAL, bg=self.panel_color, fg=self.text_color, bd=0, highlightthickness=0, troughcolor=self.bg_color, activebackground=self.accent_color, length=200)
        self.light_slider.set(1.0)
        self.light_slider.pack(pady=5)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCheckbutton", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 10))
        ttk.Checkbutton(control_frame, text=" Enable Multi-Modal AI Detection", variable=self.analyze_mode, style="TCheckbutton").pack(pady=15)
        
        tk.Button(control_frame, text="CAPTURE FRAME", command=self.capture_image, bg=self.bg_color, fg=self.accent_color, font=("Segoe UI", 10, "bold"), bd=1, relief="solid", activebackground=self.accent_color, activeforeground=self.bg_color, cursor="hand2", width=20, pady=5).pack(pady=10)
        tk.Frame(control_frame, bg="#444", height=1).pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(control_frame, text="KINEMATIC CONTROLS", font=("Segoe UI", 9, "bold"), fg="#8BE9FD", bg=self.panel_color).pack(pady=5)
        nav_text = "[W] INSERT\n[S] RETRACT\n[A] CURL LEFT\n[D] CURL RIGHT"
        tk.Label(control_frame, text=nav_text, fg="#AAAAAA", bg=self.panel_color, font=("Consolas", 10), justify=tk.LEFT).pack(pady=5)
        
        self.root.bind('<KeyPress>', self.handle_keypress)
        self.update_video_feed()

    def handle_keypress(self, event):
        key = event.keysym.lower()
        step = 0.05
        
        if key == 'w': 
            self.insertion_pos = min(self.insertion_pos + step, 1.5)
        elif key == 's': 
            self.insertion_pos = max(self.insertion_pos - step, -0.5)
        elif key == 'a':
            # Gradual Curl Math: Tip bends most, mid bends moderately, base bends slightly
            self.distal_pos = min(self.distal_pos + step, 1.57)
            self.mid_pos = min(self.mid_pos + (step * 0.7), 1.0)
            self.proximal_pos = min(self.proximal_pos + (step * 0.4), 0.6)
        elif key == 'd':
            self.distal_pos = max(self.distal_pos - step, -1.57)
            self.mid_pos = max(self.mid_pos - (step * 0.7), -1.0)
            self.proximal_pos = max(self.proximal_pos - (step * 0.4), -0.6)
            
        msg = Float64MultiArray()
        # MUST match the order in controllers.yaml exactly!
        msg.data = [self.insertion_pos, self.proximal_pos, self.mid_pos, self.distal_pos]
        self.cmd_pub.publish(msg)

    def image_callback(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.current_frame = frame.copy()

    def capture_image(self):
        filename = f"broncho_capture_{datetime.now().strftime('%H%M%S')}.png"
        if self.analyze_mode.get():
            save_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_RGB2BGR)
            prefix = "[AI-PROCESSED]"
        else:
            save_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            prefix = "[RAW]"
        cv2.imwrite(filename, save_frame)
        print(f"{prefix} Image saved: {filename}")

    def add_medical_hud(self, frame):
        h, w = frame.shape[:2]
        cv2.circle(frame, (w-60, 30), 6, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w-45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"LIVE | {timestamp}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        telemetry = f"INS: {self.insertion_pos:.2f}m | PROX: {self.proximal_pos:.2f}r | DIST: {self.distal_pos:.2f}r"
        cv2.putText(frame, telemetry, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 216), 1, cv2.LINE_AA)
        cx, cy = int(w/2), int(h/2)
        cv2.line(frame, (cx-10, cy), (cx+10, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy-10), (cx, cy+10), (255, 255, 255), 1)
        return frame

    def advanced_ai_pipeline(self, frame):
        # 1. Advanced Noise Reduction (Bilateral Filter preserves structural edges better than Median)
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 2. CLAHE Contrast Normalization
        lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
        
        # 3. HSV Conversion for Multi-Spectral Masking
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        
        # Define accurate Spectral Ranges for the 5 Anomaly Classes
        m_red1 = cv2.inRange(hsv, np.array([0, 100, 40]), np.array([10, 255, 255]))
        m_red2 = cv2.inRange(hsv, np.array([170, 100, 40]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(m_red1, m_red2)
        
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 50]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([35, 50, 40]), np.array([85, 255, 255]))
        mask_blue = cv2.inRange(hsv, np.array([90, 80, 40]), np.array([130, 255, 255]))
        
        # Low Value (Brightness) detector for Necrotic/Black masses
        mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
        
        # Dictionary linking masks to AI labels and BGR bounding box colors
        anomaly_classes = [
            ("RED", mask_red, (0, 0, 255)),     
            ("YELLOW", mask_yellow, (0, 255, 255)), 
            ("GREEN", mask_green, (0, 255, 0)),   
            ("BLUE", mask_blue, (255, 0, 0)),    
            ("DARK", mask_dark, (200, 0, 200))    
        ]
        
        kernel = np.ones((5,5), np.uint8)
        
        # 4. Feature Extraction & Expert Logic System
        for color_name, raw_mask, box_color in anomaly_classes:
            # Clean acoustic noise from the mask
            clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100: # Size Threshold
                    
                    # Mathematical Shape Extraction
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area if hull_area > 0 else 0
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    label = "UNKNOWN ANOMALY"
                    
                    # Logic Classification Network
                    if color_name == "RED":
                        if solidity > 0.85:
                            label = f"HEMORRHAGE (Sol:{solidity:.2f})"
                        else:
                            label = f"CARCINOMA (Sol:{solidity:.2f})"
                    elif color_name == "YELLOW":
                        label = f"LIPOMA (Circ:{circularity:.2f})"
                    elif color_name == "GREEN":
                        label = f"MUCUS PLUG (Area:{area})"
                    elif color_name == "BLUE":
                        label = f"FOREIGN BODY (Capsule)"
                    elif color_name == "DARK":
                        label = f"NECROTIC MASS (Low Val)"

                    # Draw Classification UI
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
                    
                    # Draw Convex Hull to show the shape segmentation
                    cv2.drawContours(frame, [hull], -1, (255,255,255), 1)

        return frame

    def process_pipeline(self, frame):
        alpha = self.light_slider.get()
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        
        if self.analyze_mode.get():
            frame = self.advanced_ai_pipeline(frame)
            
        frame = self.add_medical_hud(frame)
        return frame

    def update_video_feed(self):
        processed = self.process_pipeline(self.current_frame)
        self.processed_frame = processed.copy()
        
        img = PILImage.fromarray(processed)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        self.root.after(30, self.update_video_feed)

def main(args=None):
    rclpy.init(args=args)
    ui_node = SmartEndoscopeUI()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ui_node,), daemon=True)
    spin_thread.start()
    ui_node.root.mainloop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import cv2
import numpy as np

class VisionProcessor:
    def __init__(self):
        """
        Initializes the static components of the AI detection pipeline.
        """
        # Step 2: Contrast enhancement setup (CLAHE: clip 3.0, 8x8 tile)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Step 4: Morphological cleaning setup (5x5 kernel)
        self.kernel = np.ones((5, 5), np.uint8)

    def process_frame(self, frame):
        """
        Passes a single BGR frame through the 7-step AI detection pipeline.
        Returns the annotated frame.
        """
        # Get screen dimensions for our "Area Ceiling" and Debugger
        screen_h, screen_w = frame.shape[:2]
        screen_area = screen_h * screen_w

        # 1. Noise reduction — Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)

        # 2. Contrast enhancement — CLAHE in LAB color space
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. HSV multi-spectral masking
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # Define HSV bands (OpenCV H is 0-180, S/V are 0-255)
        # Red: 0-10 and 170-180 (Using the more forgiving 30 saturation/value limits)
        # ULTRA-STRICT RED: Keep Saturation > 240 to ignore the wall.
        # Drop Value to 200 to allow the sphere through!
        mask_red1 = cv2.inRange(hsv, np.array([0, 240, 200]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 240, 200]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Yellow: 15-35
        mask_yellow = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))

        # Green: 35-85
        mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        # Blue: 90-130
        mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))

        # Dark: Value < 30
        mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))

        masks = {
            'Red': mask_red,
            'Yellow': mask_yellow,
            'Green': mask_green,
            'Blue': mask_blue,
            'Dark': mask_dark
        }

        output_frame = enhanced.copy()

        # Process each color band
        for color_name, mask in masks.items():
            
            # 4. Morphological cleaning — 5x5 open kernel
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            # 5. Feature extraction per contour
            # Change from RETR_EXTERNAL to RETR_LIST
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area > 100 px^2 AND less than 30% of the screen!
                if 100 < area < (screen_area * 0.30):
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)

                    # Prevent division by zero
                    if hull_area == 0:
                        continue

                    # Calculate solidity: area / hull_area
                    solidity = float(area) / hull_area

                    # 6. Classification logic
                    label = "Unknown"
                    box_color = (255, 255, 255) # Default white

                    if color_name == 'Red':
                        if solidity > 0.85:
                            label = "Hemorrhage"
                            box_color = (0, 0, 255) # BGR Red
                        else:
                            label = "Carcinoma"
                            box_color = (130, 0, 130) # Dark Purple
                    elif color_name == 'Yellow':
                        label = "Lipoma"
                        box_color = (0, 255, 255) # BGR Yellow
                    elif color_name == 'Green':
                        label = "Mucus plug"
                        box_color = (0, 255, 0) # BGR Green
                    elif color_name == 'Blue':
                        label = "Necrotic mass"  # Swapped to match the new blue model!
                    elif color_name == 'Dark':
                        label = "Necrotic mass"
                        box_color = (50, 50, 50) # Dark Grey
                    elif color_name == 'Blue':
                        label = "Foreign body"
                        box_color = (255, 0, 0) # BGR Blue

                    # 7. Output: Bounding boxes + labels + convex hull contours
                    cv2.rectangle(output_frame, (x, y), (x + w_box, y + h_box), box_color, 2)
                    cv2.drawContours(output_frame, [hull], 0, (255, 255, 255), 1) # White overlay for texture
                    
                    # Add label and solidity metric to the HUD
                    text = f"{label} (Sol: {solidity:.2f})"
                    cv2.putText(output_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # --- THE CROSSHAIR DEBUGGER ---
        # Find the center of the screen
        cx, cy = screen_w // 2, screen_h // 2
        
        # Draw a small crosshair in the center
        cv2.drawMarker(output_frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        # Read the exact HSV value of the pixel right under the crosshair
        center_hsv = hsv[cy, cx]
        
        # Print that HSV value on the bottom-left of the camera feed
        cv2.putText(output_frame, f"Center HSV: {center_hsv}", (10, screen_h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output_frame

# --- Standalone Testing Script ---
if __name__ == "__main__":
    # Initialize the processor
    vp = VisionProcessor()
    
    # Open default webcam (0) to test the pipeline live
    cap = cv2.VideoCapture(0)
    
    print("Starting AI Pipeline Test. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run the pipeline
        processed_frame = vp.process_frame(frame)
        
        # Display the output
        cv2.imshow("Smart Bronchoscope - AI Detection Test", processed_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import os
import pickle
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue
from handDetectorMo import Machine, HandDetector, SimpleHorizontalMotionFilter ,Dashboard,ONNXHandDetector

# Configuration variables
showFps = True
drawbox = True
device = 0
confidence_threshol = 0.5
motion_sensibility = 25
motion_scale = 0.25
frame_skipper = 6
yoloPath = "intern/hand_detector/yolo-Weights/best_11_480_openvino_model/best_11_480.xml"



class Application:
    def __init__(self, video_path, model_path, frame_skipper, Fps):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use default camera for demo
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.model = HandDetector(model_path, conf_thresh=confidence_threshol, draw=drawbox)
        self.detector = ONNXHandDetector(
            model_path="intern/hand_detector/yolo-Weights/inference_model_rf_detr_batch2.onnx",
            class_names=["background", "arm", "palm"],
            conf_threshold=0.3,
            input_size=384,
            draw=True
        )
        self.motion_filter = SimpleHorizontalMotionFilter()
        self.machine = Machine()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.skipper = frame_skipper
        self.j = frame_skipper
        self.Fps = Fps
        self.alternate = 0
        self.running = False
        self.show_feedback = False
        self.image_queue = queue.Queue(maxsize=1)

    def resize_with_pad(self, image, new_shape, padding_color=(0, 0, 0)):
        original_shape = (image.shape[1], image.shape[0])
        ratio_w = new_shape[0] / original_shape[0]
        ratio_h = new_shape[1] / original_shape[1]
        ratio = min(ratio_w, ratio_h)
        new_size = (int(original_shape[0] * ratio), int(original_shape[1] * ratio))
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    def resize_to_fixed_height(self, img, target_height):
        original_height, original_width = img.shape[:2]
        aspect_ratio = original_width / original_height
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(
            img,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        return resized_img

    def run(self):
        # For demo purposes, we'll simulate the calibration file existence
        CALIBRATION_FILE = 'output/calibration_data.pkl'
        print("Starting application.")
        
        # Create demo directory and file if they don't exist
        os.makedirs('output', exist_ok=True)
        if not os.path.exists(CALIBRATION_FILE):
            # Create a dummy calibration file for demo
            calibration_data = {
                'camera_matrix': np.eye(3),
                'distortion_coefficients': np.zeros(5)
            }
            with open(CALIBRATION_FILE, 'wb') as f:
                pickle.dump(calibration_data, f)
        
        # Load calibration data
        with open(CALIBRATION_FILE, 'rb') as f:
            calibration_data = pickle.load(f)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")
        
        mtx = calibration_data['camera_matrix']
        dist = calibration_data['distortion_coefficients']
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
        
        # Create undistortion maps
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame")
                break
                
            # Apply undistortion
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            undistorted = undistorted[y:y + h, x:x + w]
            frame = cv2.resize(undistorted, (width, height))
            
            horiz = frame
            try:
                motion_detected = self.motion_filter.detect_horizontal_motion(horiz)
                hand_detected = False
                
                if not motion_detected:
                    if self.j == int((self.skipper / 2)):
                        frame = self.resize_to_fixed_height(frame, 480)
                        (left_frame, right_frame), hand_detected = self.model.find(frame)
                        final = np.concatenate((left_frame, right_frame), axis=1)
                        self.j += 1
                    elif self.j == self.skipper:
                        frame = self.resize_to_fixed_height(frame, 384)
                        final, hand_detected = self.detector.detect(frame)
                        self.j = 0
                    else:
                        self.j += 1
                else:
                    final = frame
                    cv2.putText(final, "HORIZONTAL MOTION", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 6)

                if motion_detected or hand_detected:
                    self.machine.turn_off()
                else:
                    if not self.machine.should_stay_off():
                        self.machine.turn_on()
                
                self.machine.update_display(final)
                self.frame_count += 1
                
                if self.frame_count >= 10:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                    self.start_time = time.time()
                    self.frame_count = 0
                
                if self.Fps:
                    final = self.resize_to_fixed_height(final, 384)
                    cv2.putText(final, f"FPS: {int(self.fps)}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Resize final to consistent size with padding
                final = self.resize_with_pad(final, (800, 450))

                # Put final image into queue for display
                final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(final_rgb)
                try:
                    self.image_queue.put_nowait(img)
                except queue.Full:
                    pass  # Skip if queue is full

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

            time.sleep(0.01)  # Small delay

        self.cap.release()



if __name__ == "__main__":
    app = Application(device, yoloPath, frame_skipper, showFps)
    dashboard = Dashboard(app)
    dashboard.root.mainloop()
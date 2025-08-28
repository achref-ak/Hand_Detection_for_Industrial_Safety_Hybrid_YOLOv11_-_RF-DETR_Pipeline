import cv2
import numpy as np
import time
from openvino.runtime import Core
import cv2
import time
import numpy as np
import supervision as sv

from typing import Tuple, List
import onnxruntime as ort
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue
from gpiozero import OutputDevice  # Relay control for Raspberry Pi



class ONNXHandDetector:
    def __init__(self, model_path: str, class_names: List[str] = ["background","arm", "palm"], 
                 conf_threshold: float = 0.5, input_size: int = 384, 
                  draw: bool = True):

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.draw = draw

        
        # Load ONNX model
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Visualization setup
        self.palette = sv.ColorPalette.from_hex(["#FF0000", "#00FF00"])
        self.bbox_annotator = sv.BoxAnnotator(color=self.palette, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=self.palette, text_color=sv.Color.BLACK, text_scale=0.5, smart_position=True
        )
        
        # Performance tracking
        self.fps_times = []
        self.last_inference_time = 0
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process individual frame using grayscale Canny pipeline"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        # Quantization (32 levels)
        num_levels = 32
        lut = np.arange(256, dtype=np.uint8)
        lut = (lut // (256 // num_levels)) * (256 // num_levels)
        quantized = cv2.LUT(equalized, lut)
        
        # Smoothing
        blurred = cv2.boxFilter(quantized, -1, (7, 7))
        
        # Edge detection
        edges = cv2.Canny(blurred, 70, 120)
        
        # Apply edges to image
        result = blurred.copy()
        np.putmask(result, edges > 0, 0)  # Set edges to black
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def resize_with_pad(self, image: np.ndarray, new_shape: Tuple[int, int], padding_color=(0, 0, 0)):
        """Resize image with padding to maintain aspect ratio"""
        h_orig, w_orig = image.shape[:2]
        target_w, target_h = new_shape
        ratio = min(target_w / w_orig, target_h / h_orig)
        new_w, new_h = int(w_orig * ratio), int(h_orig * ratio)
        resized = cv2.resize(image, (new_w, new_h))
        
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, dy, target_h - new_h - dy, dx, target_w - new_w - dx,
            cv2.BORDER_CONSTANT, value=padding_color
        )
        return padded
    
    def rescale_boxes_from_padded_to_original_xyxy(self, boxes, ratio, dx, dy, orig_shape):
        """Rescale boxes from padded coordinates to original image coordinates"""
        out = boxes.copy().astype(np.float32)
        out[:, [0, 2]] = (out[:, [0, 2]] - dx) / ratio
        out[:, [1, 3]] = (out[:, [1, 3]] - dy) / ratio
        
        h, w = orig_shape
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
        return out
    
    def cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert center coordinates to corner coordinates"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)
    

    
    def prepare_batch(self, halves: List[np.ndarray]):
        """Prepare batch of images for inference"""
        batch_input = []
        for half in halves:
            processed = self.process_frame(half)
            padded= self.resize_with_pad(
                processed, (self.input_size, self.input_size)
            )
            inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            inp = inp.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_input.append(inp)
            
            
        return np.stack(batch_input, axis=0)
    
    def postprocess_half(self, boxes, scores):
        """Postprocess detections for a single half-frame"""
        if boxes.size == 0:
            return sv.Detections.empty(), []
            
        boxes_xyxy = self.cxcywh_to_xyxy(boxes.astype(np.float32))
        
        # Scale boxes if normalized
        if np.nanmax(boxes_xyxy) <= 1.5:
            boxes_xyxy *= float(self.input_size)
            
        
        class_ids = np.argmax(scores, axis=-1).astype(np.int32)
        confs = np.max(scores, axis=-1).astype(np.float32)
        mask = confs >= self.conf_threshold
        
        boxes_final = boxes_xyxy[mask]
        class_final = class_ids[mask]
        conf_final = confs[mask]
        
        if len(boxes_final) > 0:
            x1 = np.minimum(boxes_final[:, 0], boxes_final[:, 2])
            y1 = np.minimum(boxes_final[:, 1], boxes_final[:, 3])
            x2 = np.maximum(boxes_final[:, 0], boxes_final[:, 2])
            y2 = np.maximum(boxes_final[:, 1], boxes_final[:, 3])
            boxes_final = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            
        det = sv.Detections(
            xyxy=boxes_final if len(boxes_final) > 0 else np.zeros((0, 4), dtype=np.float32),
            class_id=class_final if len(boxes_final) > 0 else np.zeros((0,), dtype=np.int32),
            confidence=conf_final if len(boxes_final) > 0 else np.zeros((0,), dtype=np.float32)
        )
        
        labels = [
            f"{self.class_names[cid] if 0 <= cid < len(self.class_names) else 'unk'} {conf:.2f}"
            for cid, conf in zip(det.class_id, det.confidence)
        ]
        
        return det, labels
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[sv.Detections]]:

        h, w = frame.shape[:2]
        half_width = int(w // 2)
        self.shift=int(h-half_width)
      
        
        # Split into left/right halves
        left_half = frame[:, :half_width + self.shift]
        right_half = frame[:, half_width - self.shift:]
        
        # Split frame into halves with overlap
        
        
        # Prepare batch
        batch_inp = self.prepare_batch([left_half, right_half])
        
        # Run inference
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: batch_inp})
        self.last_inference_time = (time.time() - t0) * 1000.0  # ms
        
        boxes_batch, scores_batch = outputs
        
        # Postprocess each half
        left_det, left_labels = self.postprocess_half(boxes_batch[0], scores_batch[0])
        right_det, right_labels = self.postprocess_half(boxes_batch[1], scores_batch[1])
        
        hand_detected = len(left_det.xyxy) > 0 or len(right_det.xyxy) > 0
        # Draw annotations if enabled
        if self.draw:
            left_annot = self.bbox_annotator.annotate(left_half.copy(), left_det)
            left_annot = self.label_annotator.annotate(left_annot, left_det, left_labels)
            
            right_annot = self.bbox_annotator.annotate(right_half.copy(), right_det)
            right_annot = self.label_annotator.annotate(right_annot, right_det, right_labels)
            
            left_annot =left_annot[:, :self.input_size-self.shift]
                    
            right_annot =right_annot[:,self.shift:]
            
            annotated_frame = np.hstack([left_annot, right_annot])
            
            # Add performance info
            now = time.time()
            self.fps_times.append(now)
            if len(self.fps_times) > 10:
                self.fps_times.pop(0)
                
            """avg_fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0]) if len(self.fps_times) >= 2 else 0.0
            
            cv2.putText(
                annotated_frame, 
                f"FPS(10-avg): {avg_fps:.1f} | Inference: {self.last_inference_time:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )"""
        else:
            annotated_frame = frame.copy()
            
        return annotated_frame,hand_detected
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "last_inference_ms": self.last_inference_time,
            "avg_fps": len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0]) if len(self.fps_times) >= 2 else 0
        }
    

    
class HandDetectorTransformers:
    def __init__(self, model_path, conf_thresh=0.5, input_size=640, slice_wh=(640, 640), draw=True):
        self.classNames = ["arm", "palm"]
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.showAnn = draw
        self.slice_wh = slice_wh

        # -------------------------------
        # Load RF-DETR model
        # -------------------------------
        print("Loading RF-DETR Nano model...")
        self.model = RFDETRNano(pretrain_weights=model_path)
        self.model.optimize_for_inference()

        # -------------------------------
        # SAHI slicer setup
        # -------------------------------
        self.slicer = sv.InferenceSlicer(
            callback=self._callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=self.slice_wh
        )

        # Visualization tools
        color_palette = sv.ColorPalette.from_hex(["#FF0000", "#00FF00"])
        self.bbox_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=color_palette,
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            smart_position=True
        )

    def process_frame(self, frame):
        """Apply grayscale + equalization + quantization + smoothing + edge filtering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        num_levels = 32
        lut = np.arange(256, dtype=np.uint8)
        lut = (lut // (256 // num_levels)) * (256 // num_levels)
        quantized = cv2.LUT(equalized, lut)
        blurred = cv2.boxFilter(quantized, -1, (7, 7))
        edges = cv2.Canny(blurred, 70, 120)
        result = blurred.copy()
        np.putmask(result, edges > 0, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def _callback(self, image_slice: np.ndarray) -> sv.Detections:
        """Callback used by SAHI slicer"""
        input_image = cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(input_image, threshold=self.conf_thresh)
        if len(detections.class_id) > 0:
            detections.class_id = np.array([int(c) for c in detections.class_id])
        return detections

    def find(self, image):
        """Run inference with slicing and return annotated image + detection flag"""
        frame_proc = self.process_frame(image)
        detections = self.slicer(frame_proc)

        hand_detected = False
        labels = []
        for cid in detections.class_id:
            cid_corrected = cid - 1
            if 0 <= cid_corrected < len(self.classNames):
                labels.append(f"{self.classNames[cid_corrected]}")
                hand_detected = True
            else:
                labels.append("unknown")

        if self.showAnn:
            annotated_frame = self.bbox_annotator.annotate(image.copy(), detections)
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        else:
            annotated_frame = image

        return annotated_frame, hand_detected


import cv2
import time
from gpiozero import OutputDevice  # Relay control for Raspberry Pi

class Machine:
    def __init__(self, relay_pin=17):  # GPIO17 default
        self.is_on = False
        self.last_off_time = 0
        self.last_state = "off"

        # Relay setup (active_high=True means GPIO.HIGH = relay ON)
        self.relay = OutputDevice(relay_pin, active_high=True, initial_value=False)

    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            self.last_state = "on"
            self.relay.on()  # Activate relay

    def turn_off(self):
        if self.is_on:
            self.is_on = False
            self.last_off_time = time.time()
            self.last_state = "off"
            self.relay.off()  # Deactivate relay

    def should_stay_off(self):
        return (time.time() - self.last_off_time) < 2.0

    def update_display(self, image):
        if self.is_on or (not self.is_on and self.last_state == "on" and self.should_stay_off()):
            cv2.putText(image, "process continued", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "process paused", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def cleanup(self):
        self.relay.off()  # Make sure relay is OFF before exit

class HandDetector:
    def __init__(self, model_path, device="CPU", conf_thresh=0.5, nms_thresh=0.3, input_size=640,skipper=5,draw=True):
        self.classNames = ["arm", "palm"]
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.skipper = skipper
        self.j = skipper
        self.showAnn=draw
        # Load OpenVINO model
        ie = Core()
        model = ie.read_model(model=model_path)
        input_layer = model.input(0)
        input_layer_shape = input_layer.partial_shape
        input_layer_shape[0] = 2  # Set batch size to 2
        model.reshape({input_layer: input_layer_shape})
        self.compiled = ie.compile_model(model, device_name=device)
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)

    def process_frame(self, frame):
        """Process individual frame using grayscale Canny pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Quantization (32 levels)
        num_levels = 32
        lut = np.arange(256, dtype=np.uint8)
        lut = (lut // (256//num_levels)) * (256//num_levels)
        quantized = cv2.LUT(equalized, lut)
        
        # Smoothing
        blurred = cv2.boxFilter(quantized, -1, (7, 7))
        
        # Edge detection
        edges = cv2.Canny(blurred, 70, 120)
        
        # Apply edges to image
        result = blurred.copy()
        np.putmask(result, edges > 0, 0)  # Set edges to black
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def resize_pad(self, image, size=(640, 640)):
     """Resize and pad image to square"""
     if size is None:
        size = (self.input_size, self.input_size)

     h, w = image.shape[:2]
     target_h, target_w = size
     scale = min(target_h / h, target_w / w)

     nh, nw = int(h * scale), int(w * scale)
     img_resized = cv2.resize(image, (nw, nh))
    
     top = (target_h - nh) // 2
     bottom = target_h - nh - top
     left = (target_w - nw) // 2
     right = target_w - nw - left

     img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
     return img_padded


    def preprocess_batch(self, images):
        """Preprocess batch of images"""
        blobs = []
        for img in images:
            img_resized = self.resize_pad(img)
            blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
            blobs.append(blob[0])  # Remove batch dimension     
        return np.stack(blobs, axis=0)


    def postprocess(self, pred, orig_size=640):
        """Decode predictions (without NMS)"""
        boxes, scores, ids = [], [], []
        
        # Handle different output formats
        if len(pred.shape) == 3 and pred.shape[0] == 1:
            if pred.shape[1] == 6:
                pred = pred[0].T
            elif pred.shape[2] == 6:
                pred = pred[0]
        
        # Apply sigmoid if needed
        class_confs = pred[:, 4:]
        
        # Extract boxes
        for i in range(pred.shape[0]):
            scores_for_detection = class_confs[i]
            max_score = np.max(scores_for_detection)
            class_id = np.argmax(scores_for_detection)
            
            if max_score > self.conf_thresh:
                x_center, y_center, width, height = pred[i, :4]
                x1 = x_center - width/2.0
                y1 = y_center - height/2.0
                w = width
                h = height
                
                # Clip to padded image boundaries
                x1 = max(0.0, min(x1, orig_size))
                y1 = max(0.0, min(y1, orig_size))
                w = min(w, orig_size - x1)
                h = min(h, orig_size - y1)
                
                if w > 0 and h > 0:
                    boxes.append([x1, y1, w, h])
                    scores.append(float(max_score))
                    ids.append(int(class_id))
        
        return boxes, scores, ids

    def find(self, image):
        h, w = image.shape[:2] 
        half_width = int(w // 2)
        self.shift=int(h-half_width)
        print(self.shift)
        # Apply image processing pipeline
        
        
        # Split into left/right halves
        left_img = image[:, :half_width + self.shift]
        right_img = image[:, half_width - self.shift:]
        images = [left_img, right_img]
        origi=images.copy()
        origi[0] = self.process_frame(origi[0])
        origi[1] = self.process_frame(origi[1])
        
      
        # Preprocess batch
        batch_blob = self.preprocess_batch(origi)
        
        # Run inference
        preds = self.compiled([batch_blob])[self.output_layer]
        
        # Process results
        hand_detected = False
        for i in range(2):
            pred = preds[i:i+1]  # Add batch dimension
            boxes, scores, ids = self.postprocess(pred)
            
            # Apply global NMS
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
                if len(indices) > 0:
                    indices = indices.flatten()
                    for idx in indices:
                        x, y, w_box, h_box = boxes[idx]  # Float coordinates in padded image
                        cls = ids[idx]
                        label = self.classNames[cls]
                        conf = scores[idx]
                        
                        # Get original half-image dimensions
                        orig_h, orig_w = images[i].shape[:2]
                        
                        # Compute scale and padding for this half-image
                        scale_val = min(self.input_size / orig_h, self.input_size / orig_w)
                        nh = int(orig_h * scale_val)
                        nw = int(orig_w * scale_val)
                        top_pad = (self.input_size - nh) // 2
                        left_pad = (self.input_size - nw) // 2
                        
                        # Adjust box coordinates to original half-image
                        x_adj = (x - left_pad) / scale_val
                        y_adj = (y - top_pad) / scale_val
                        w_adj = w_box / scale_val
                        h_adj = h_box / scale_val
                        
                        # Clip to original image boundaries
                        x_adj = max(0, min(orig_w, x_adj))
                        y_adj = max(0, min(orig_h, y_adj))
                        w_adj = min(w_adj, orig_w - x_adj)
                        h_adj = min(h_adj, orig_h - y_adj)
                        
                        # Convert to integers for drawing
                        x_adj, y_adj, w_adj, h_adj = int(x_adj), int(y_adj), int(w_adj), int(h_adj)
                        
                        # Skip if box has zero area
                        if w_adj <= 0 or h_adj <= 0:
                            continue
                        
                        # Adjust coordinates to full image if needed
                       
                        
                        # Clip to full image boundaries
                        x_adj = max(0, min(w, x_adj))
                        y_adj = max(0, min(h, y_adj))
                        w_adj = min(w_adj, w - x_adj)
                        h_adj = min(h_adj, h - y_adj)
                        if(self.showAnn):
                         cv2.rectangle(images[i], (x_adj, y_adj), (x_adj + w_adj, y_adj + h_adj), (0, 255, 0), 2)
                         cv2.putText(images[i], f"{label} {conf:.2f}", (x_adj, y_adj - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                        if label in ["arm", "palm"]:
                         cv2.putText(images[i], "HAND DETECTED", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                         hand_detected = True
            if not hand_detected:
             cv2.putText(images[i], "NO HANDS DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          
        images[0] = images[0][:, :640-self.shift]
                    
        images[1] = images[1][:,self.shift:]
        
            
        return images, hand_detected

class SimpleHorizontalMotionFilter:
    def __init__(self, sensitivity=50, horizontal_bias=1.5, resize_factor=0.5):
        self.sensitivity = sensitivity
        self.horizontal_bias = horizontal_bias
        self.prev_frame = None
        self.resize_factor = resize_factor

    def preprocess(self, frame):
        if self.resize_factor < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.blur(gray, (5, 5))

    def detect_horizontal_motion(self, frame):
        gray = self.preprocess(frame)
        if self.prev_frame is None:
            self.prev_frame = gray
            return frame, False
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_frame = gray
        for cnt in contours:
            if cv2.contourArea(cnt) < 400:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w * self.horizontal_bias:
                return  True
        return  False


class Dashboard:
    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Hand Safety Monitoring System")
        
        # Remove window decorations and make fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#f0f0f0")
        
        # Get the actual screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        print(f"Screen resolution: {screen_width}x{screen_height}")

        # Configure styles with smaller fonts
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors with smaller fonts
        self.style.configure('Title.TLabel', background='#2c3e50', foreground='white', 
                            font=('Arial', 14, 'bold'))
        self.style.configure('Control.TFrame', background='#ecf0f1')
        self.style.configure('Status.TFrame', background='#34495e')
        self.style.configure('Control.TButton', font=('Arial', 8, 'bold'),
                            background='#3498db', foreground='white')
        self.style.configure('Toggle.TButton', font=('Arial', 8, 'bold'),
                            background='#e74c3c', foreground='white')
        self.style.configure('Exit.TButton', font=('Arial', 8, 'bold'),
                            background='#e74c3c', foreground='white')
        self.style.configure('Status.TLabel', background='#34495e', foreground='white', 
                            font=('Arial', 8))
        self.style.configure('Fps.TLabel', background='#34495e', foreground='#2ecc71', 
                            font=('Arial', 8, 'bold'))
        self.style.configure('StatusIndicator.Running.TLabel', background='#2ecc71', 
                            foreground='white', font=('Arial', 8, 'bold'))
        self.style.configure('StatusIndicator.Stopped.TLabel', background='#e74c3c', 
                            foreground='white', font=('Arial', 8, 'bold'))

        # Calculate proportional heights based on screen size
        title_height = int(screen_height * 0.07)  # 7% of screen height
        video_height = int(screen_height * 0.65)  # 65% of screen height
        control_height = int(screen_height * 0.18)  # 18% of screen height
        status_height = int(screen_height * 0.10)  # 10% of screen height

        # Create main frames with proportional heights
        self.title_frame = ttk.Frame(self.root, style='Title.TLabel', height=title_height)
        self.title_frame.pack(fill=tk.X, pady=(0, 2))
        
        self.video_frame = ttk.Frame(self.root, height=video_height)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)
        self.video_frame.pack_propagate(False)
        
        self.control_frame = ttk.Frame(self.root, style='Control.TFrame', height=control_height)
        self.control_frame.pack(fill=tk.X, padx=10, pady=2)
        self.control_frame.pack_propagate(False)
        
        self.status_frame = ttk.Frame(self.root, style='Status.TFrame', height=status_height)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)

        # Title
        title_label = ttk.Label(self.title_frame, text="HAND SAFETY MONITORING SYSTEM", 
                               style='Title.TLabel')
        title_label.pack(expand=True)

        # Video display
        self.label = ttk.Label(self.video_frame)
        self.label.pack(expand=True)

        # Control buttons
        control_title = ttk.Label(self.control_frame, text="System Controls", 
                                 font=('Arial', 10, 'bold'), background='#ecf0f1')
        control_title.grid(row=0, column=0, columnspan=4, pady=(2, 2))

        self.start_btn = ttk.Button(self.control_frame, text="Start",
                                   command=self.start, style='Control.TButton')
        self.start_btn.grid(row=1, column=0, padx=2, pady=1, sticky="ew")

        self.stop_btn = ttk.Button(self.control_frame, text="Stop",
                                  command=self.stop, style='Control.TButton')
        self.stop_btn.grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        self.toggle_btn = ttk.Button(self.control_frame, text="Video: OFF",
                                    command=self.toggle_feedback, style='Toggle.TButton')
        self.toggle_btn.grid(row=1, column=2, padx=2, pady=1, sticky="ew")

        self.exit_btn = ttk.Button(self.control_frame, text="Exit",
                                  command=self.exit_app, style='Exit.TButton')
        self.exit_btn.grid(row=1, column=3, padx=2, pady=1, sticky="ew")

        # Add fullscreen toggle button (optional)
        self.fullscreen_btn = ttk.Button(self.control_frame, text="Window",
                                        command=self.toggle_fullscreen, style='Control.TButton')
        self.fullscreen_btn.grid(row=2, column=0, columnspan=4, padx=2, pady=1, sticky="ew")

        # Configure grid weights for responsive layout
        for i in range(4):
            self.control_frame.columnconfigure(i, weight=1)

        # Status bar
        self.status_indicator = ttk.Label(self.status_frame, text="STOPPED", 
                                        style='StatusIndicator.Stopped.TLabel')
        self.status_indicator.pack(side=tk.LEFT, padx=(5, 2), pady=1)

        self.status_label = ttk.Label(self.status_frame, text="Status: Stopped",
                                     style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=(0, 5), pady=1)

        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0", style='Fps.TLabel')
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=1)

        self.info_label = ttk.Label(self.status_frame, text="v1.0",
                                   style='Status.TLabel')
        self.info_label.pack(side=tk.RIGHT, padx=5, pady=1)

        # Initialize
        self.thread = None
        self.is_fullscreen = True
        
        # Calculate image size based on video frame size
        video_width = screen_width - 20  # account for padding
        video_frame_height = video_height
        
        # Load your custom placeholder image
        try:
            self.placeholder_img = Image.open("placeholder.jpg")
            self.placeholder_img = self.placeholder_img.resize((video_width, video_frame_height), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading placeholder image: {e}")
            # Fallback to a simple text image
            self.placeholder_img = Image.new('RGB', (video_width, video_frame_height), (240, 240, 240))
            draw = ImageDraw.Draw(self.placeholder_img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            text = "Video Feed Disabled"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((video_width - text_width) // 2, (video_frame_height - text_height) // 2)
            draw.text(position, text, fill=(100, 100, 100), font=font)
        
        self.current_img = self.placeholder_img
        self.video_width = video_width
        self.video_height = video_frame_height
        
        self.update_image()
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        # Add Escape key to exit fullscreen
        self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        self.fullscreen_btn.config(text="Window" if self.is_fullscreen else "Fullscreen")

    def start(self):
        if not self.app.running:
            self.app.running = True
            self.status_label.config(text="Status: Active")
            self.status_indicator.config(text="RUNNING", style='StatusIndicator.Running.TLabel')
            self.thread = threading.Thread(target=self.app.run)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.app.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.status_label.config(text="Status: Stopped")
        self.status_indicator.config(text="STOPPED", style='StatusIndicator.Stopped.TLabel')

    def toggle_feedback(self):
        self.app.show_feedback = not self.app.show_feedback
        state = "ON" if self.app.show_feedback else "OFF"
        self.toggle_btn.config(text=f"Video: {state}")
        
        if not self.app.show_feedback:
            self.current_img = self.placeholder_img
            imgtk = ImageTk.PhotoImage(self.current_img)
            self.label.config(image=imgtk)
            self.label.image = imgtk

    def exit_app(self):
        self.stop()
        self.root.quit()
        self.root.destroy()

    def update_image(self):
        if self.app.show_feedback:
            try:
                img = self.app.image_queue.get_nowait()
                # Resize image to fit the video frame
                img = img.resize((self.video_width, self.video_height), Image.LANCZOS)
                self.current_img = img
            except queue.Empty:
                pass

        imgtk = ImageTk.PhotoImage(self.current_img)
        self.label.config(image=imgtk)
        self.label.image = imgtk
        self.fps_label.config(text=f"FPS: {int(self.app.fps)}")
        self.root.after(50, self.update_image)
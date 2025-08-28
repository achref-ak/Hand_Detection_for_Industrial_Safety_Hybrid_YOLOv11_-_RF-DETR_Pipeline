import cv2
import numpy as np
import time
import os
import pickle
from openvino.runtime import Core
from handDetectorMo import Machine, HandDetector, SimpleHorizontalMotionFilter , HandDetectorTransformers,ONNXHandDetector

showFps=True
drawbox=True
device=0
confidence_threshol=0.5
motion_sensibility=25
motion_scale=0.25
frame_skipper=6

#yoloPath="intern/hand_detector/yolo-Weights/lastbest_yolov8_640_openvino_model/lastbest.xml"
yoloPath="intern/hand_detector/yolo-Weights/best_11_480_openvino_model/best_11_480.xml"


class Application: 
    def __init__(self, video_path, model_path,frame_skipper,Fps):
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        #self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
        self.model = HandDetector(model_path,conf_thresh=confidence_threshol,draw=drawbox)
        """self.rf_detector = HandDetectorTransformers( model_path="checkpoint_best_ema.pth",
               conf_thresh=0.5,
               input_size=640,
               slice_wh=(640, 640),
               draw=True
)"""
        self.detector = ONNXHandDetector(
           model_path="intern\hand_detector\yolo-Weights\inference_model_rf_detr_batch2.onnx",
           class_names=["background","arm", "palm"],
           conf_threshold=0.3,
           input_size=384,
           
           draw=True
)
        self.motion_filter = SimpleHorizontalMotionFilter()
        self.machine = Machine()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.skipper=frame_skipper
        self.j=frame_skipper
        self.Fps=Fps
        self.alternate=0

    def resize_with_pad(self, image, new_shape, padding_color=(0, 0, 0)):
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
   
    def resize_to_fixed_height(slef,img, target_height):
    # Get original dimensions
     original_height, original_width = img.shape[:2]
    
    # Calculate new width to maintain aspect ratio
     aspect_ratio = original_width / original_height
     new_width = int(target_height * aspect_ratio)
    
    # Resize the image
     resized_img = cv2.resize(
        img, 
        (new_width, target_height),  # (width, height)
        interpolation=cv2.INTER_AREA  # Use for shrinking
    )
    
     return resized_img
   
    
    def run(self):
        CALIBRATION_FILE='output/calibration_data.pkl'
        print("Starting application. Press 'q' to quit.")
        if not os.path.exists(CALIBRATION_FILE):
         print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
         print("Please run camera_calibration.py first to generate calibration data.")
         return
    
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
        while True:
            success, frame = self.cap.read()
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            
            # Crop the image (optional)
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            
            # Resize to original size for display
            frame = cv2.resize(undistorted, (width, height))
            horiz=frame
            if not success:
                print("End of video or failed to read frame")
                break
            try:
                #motion_detected = self.motion_filter.detect_horizontal_motion(horiz)
                #to disable motion detection we comment the previois line and uncomment the next line 
                motion_detected=False
                hand_detected = False
                if not motion_detected:
                    if(self.j==int((self.skipper/2))):
                          frame=self.resize_to_fixed_height(frame, 480)
                          (left_frame, right_frame), hand_detected = self.model.find(frame)
                          final = np.concatenate((left_frame, right_frame), axis=1)
                          self.j+=1 
                    elif(self.j==self.skipper): 
                             frame=self.resize_to_fixed_height(frame, 384)
                             
                             final,hand_detected= self.detector.detect(frame)
                             self.j=0
                    else :
                     self.j+=1 
                else :
                   
                    final=frame
                    cv2.putText(final, "HORIZONTAL MOTION", (20, 10),
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
                if(self.Fps):  
                 final=self.resize_to_fixed_height(final, 384)
                 cv2.putText(final, f"FPS: {int(self.fps)}", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Result", final)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Application(device, yoloPath,frame_skipper,showFps)
    app.run()
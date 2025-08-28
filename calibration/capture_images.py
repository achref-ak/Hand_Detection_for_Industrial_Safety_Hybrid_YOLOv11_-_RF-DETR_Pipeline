import cv2
import os

# Capture parameters
CAMERA_ID = 0
CHESSBOARD_SIZE = (8, 5)
OUTPUT_DIRECTORY = 'calibration_images'

def capture_calibration_images():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    img_counter = 0
    print("Press 'c' to capture an image")
    print("Press 'q' or Escape to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # Make a copy for display (with drawings)
        display_frame = frame.copy()
        if ret_chess:
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_chess)
            cv2.putText(display_frame, "Chessboard detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {img_counter}", (50, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            # Save the original frame (no drawings)
            img_name = os.path.join(OUTPUT_DIRECTORY, f"calibration_{img_counter:02d}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            img_counter += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_counter} images for calibration")

if __name__ == "__main__":
    capture_calibration_images()

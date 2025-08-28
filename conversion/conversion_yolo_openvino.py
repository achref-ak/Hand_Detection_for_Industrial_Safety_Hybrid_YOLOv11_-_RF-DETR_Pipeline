from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("intern/hand_detector/yolo-Weights/best_11_480.pt")

# Export the model
model.export(format="openvino",half=True,dynamic=True,batch=2)  # creates 'yolo11n_openvino_model/'

# Load the exported OpenVINO model

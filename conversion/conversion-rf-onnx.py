from rfdetr import RFDETRNano

# Load model
model = RFDETRNano(pretrain_weights="checkpoint_best_ema.pth")

# Export ONNX with batch=2 and new input size 512x512
model.export(
    output_dir="output",
    filename="inference_model.onnx",
    batch_size=2,
    img_size=512,
    dynamic_input_shape=True
)

print("Model exported with batch size 2 and input size 512!")

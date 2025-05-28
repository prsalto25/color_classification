from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")


# from ultralytics import YOLO

# # Load a YOLO11n PyTorch model
# model = YOLO("yolo8n.pt")

# # Export the model to TensorRT with DLA enabled (only works with FP16 or INT8)
# model.export(format="engine", device="dla:0", half=True)  # dla:0 or dla:1 corresponds to the DLA cores

# # Load the exported TensorRT model
# trt_model = YOLO("yolo11n.engine")

# # Run inference
# results = trt_model("https://ultralytics.com/images/bus.jpg")
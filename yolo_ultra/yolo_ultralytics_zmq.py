import os
import zmq
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from collections import deque

# Simple CNN for color classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Voting system
class RobustColorVotingSystem:
    def __init__(self, buffer_size=5, confidence_threshold=0.6, stability_frames=3):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.color_history = {}
        self.stable_predictions = {}
        self.frame_count = {}

    def add_prediction(self, obj_id, color_name, confidence):
        if obj_id not in self.color_history:
            self.color_history[obj_id] = deque(maxlen=self.buffer_size)
            self.frame_count[obj_id] = 0
        self.color_history[obj_id].append((color_name, confidence))
        self.frame_count[obj_id] += 1

    def get_final_color(self, obj_id):
        if obj_id not in self.color_history or len(self.color_history[obj_id]) == 0:
            return "unknown", 0.0
        color_weights = {}
        total_weight = 0
        recent_colors = list(self.color_history[obj_id])
        for i, (color_name, confidence) in enumerate(recent_colors):
            if color_name not in ["uncertain", "unknown"]:
                temporal_weight = (i + 1) / len(recent_colors)
                conf_weight = confidence if confidence > self.confidence_threshold else 0.1
                weight = temporal_weight * conf_weight
                color_weights[color_name] = color_weights.get(color_name, 0) + weight
                total_weight += weight
        if not color_weights:
            return "unknown", 0.0
        best_color = max(color_weights.items(), key=lambda x: x[1])
        final_confidence = best_color[1] / total_weight if total_weight > 0 else 0
        if self.frame_count[obj_id] >= self.stability_frames and final_confidence > self.confidence_threshold:
            self.stable_predictions[obj_id] = (best_color[0], final_confidence)
        return self.stable_predictions.get(obj_id, (best_color[0], final_confidence))

# Gentle preprocessing
def gentle_preprocessing(image):
    try:
        filtered = cv2.bilateralFilter(image, 5, 20, 20)
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced, table)
        hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.05)
        s = np.clip(s, 0, 255).astype(np.uint8)
        balanced_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(balanced_hsv, cv2.COLOR_HSV2BGR)
    except:
        return image

# Configuration and model loading
COLOR_MODEL_PATH = "ccmodel.pth"
COLOR_CLASSES = ['beige_brown', 'black', 'blue', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
ENABLE_COLOR = True
USE_GENTLE_PREPROCESSING = False
model = YOLO("yolov8n.pt")
color_model = None
device = None
voting_system = None

if ENABLE_COLOR:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    color_model = SimpleCNN(len(COLOR_CLASSES))
    checkpoint = torch.load(COLOR_MODEL_PATH, map_location=device)
    color_model.load_state_dict(checkpoint['model_state_dict'])
    color_model.eval().to(device)
    voting_system = RobustColorVotingSystem()
    color_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Parse the shape string from bytes
def _parse_shape(shape_bytes):
    try:
        h, w, c = map(int, shape_bytes.decode().split("_"))
        return h, w, c
    except:
        return 0, 0, 0

# Parse the frame bytes into an image array
def _parse_frame(frame_bytes, h, w, c):
    try:
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, c))
    except:
        return None

# Extract center region of a crop
def _extract_center_region(crop, region_ratio=0.7):
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return crop
    h, w = crop.shape[:2]
    ch, cw = int(h * region_ratio), int(w * region_ratio)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    return gentle_preprocessing(crop[y1:y1 + ch, x1:x1 + cw]) if USE_GENTLE_PREPROCESSING else crop[y1:y1 + ch, x1:x1 + cw]

# Predict Color
def _predict_color(crop_image, obj_id=None):
    if not ENABLE_COLOR or color_model is None:
        return "unknown", 0.0
    try:
        center_crop = _extract_center_region(crop_image)
        crop_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        input_tensor = color_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = color_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        pred_color = COLOR_CLASSES[predicted_class.item()]
        conf_score = confidence.item()
        raw_color = pred_color if conf_score > 0.6 else "uncertain"
        if voting_system and obj_id:
            voting_system.add_prediction(obj_id, raw_color, conf_score)
            return voting_system.get_final_color(obj_id)
        return raw_color, conf_score
    except:
        return "unknown", 0.0

# Object ID
def _generate_object_id(box, class_id):
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return f"{int(class_id)}_{cx // 100}_{cy // 100}"

# Result 
def _process_result(result, frame):
    if not result or len(result) == 0:
        return np.array([[]], dtype=np.float32)
    pred = result[0].boxes
    if pred is None or pred.xyxy is None:
        return np.array([[]], dtype=np.float32)
    boxes = pred.xyxy.cpu().numpy()
    conf = pred.conf.cpu().numpy()
    cls = pred.cls.cpu().numpy()
    enhanced_results = []
    for box, confidence, class_id in zip(boxes, conf, cls):
        x1, y1, x2, y2 = map(int, box)
        color_name, color_conf = "unknown", 0.0
        if ENABLE_COLOR and x1 < x2 and y1 < y2:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                obj_id = _generate_object_id(box, class_id)
                color_name, color_conf = _predict_color(crop, obj_id)
        color_id = COLOR_CLASSES.index(color_name) if color_name in COLOR_CLASSES else -1
        enhanced_results.append([x1, y1, x2, y2, confidence, class_id, color_id, color_conf])
    return np.array(enhanced_results, dtype=np.float32)

# Start the ZMQ server
def main():
    port = 8891
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(f"tcp://127.0.0.1:{port}")
    while True:
        try:
            msg = socket.recv_multipart()
            frame_bytes, shape_bytes = msg
            h, w, c = _parse_shape(shape_bytes)
            if h == 0 or w == 0:
                socket.send_pyobj(np.array([[]], dtype=np.float32))
                continue
            frame = _parse_frame(frame_bytes, h, w, c)
            if frame is None:
                socket.send_pyobj(np.array([[]], dtype=np.float32))
                continue
            result = model(frame, verbose=False)
            detections = _process_result(result, frame)
            socket.send_pyobj(detections)
        except KeyboardInterrupt:
            socket.close()
            context.term()
            break
        except:
            socket.send_pyobj(np.array([[]], dtype=np.float32))

# Main
if __name__ == "__main__":
    main()

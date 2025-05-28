import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import os
from collections import deque, Counter
from datetime import datetime
import time

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

class RobustColorVotingSystem:
    """Enhanced voting system with smoothing and stability"""
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
        
        # Weighted voting with temporal smoothing
        color_weights = {}
        total_weight = 0
        recent_colors = list(self.color_history[obj_id])
        
        # Give more weight to recent predictions
        for i, (color_name, confidence) in enumerate(recent_colors):
            if color_name not in ["uncertain", "unknown"]:
                # Temporal weight: more recent = higher weight
                temporal_weight = (i + 1) / len(recent_colors)
                # Confidence weight
                conf_weight = confidence if confidence > self.confidence_threshold else 0.1
                # Combined weight
                weight = temporal_weight * conf_weight
                
                color_weights[color_name] = color_weights.get(color_name, 0) + weight
                total_weight += weight
        
        if not color_weights:
            return "unknown", 0.0
        
        # Get best color
        best_color = max(color_weights.items(), key=lambda x: x[1])
        final_confidence = best_color[1] / total_weight if total_weight > 0 else 0
        
        # Stability check: only update stable prediction if we have enough evidence
        if (self.frame_count[obj_id] >= self.stability_frames and 
            final_confidence > self.confidence_threshold):
            self.stable_predictions[obj_id] = (best_color[0], final_confidence)
        
        # Return stable prediction if available, otherwise current best
        if obj_id in self.stable_predictions:
            return self.stable_predictions[obj_id]
        else:
            return best_color[0], final_confidence
    
    def get_confidence_status(self, obj_id):
        """Get confidence status for color coding"""
        _, confidence = self.get_final_color(obj_id)
        if confidence >= 0.8:
            return "HIGH", (0, 255, 0)  # Green
        elif confidence >= 0.6:
            return "MEDIUM", (0, 255, 255)  # Yellow
        else:
            return "LOW", (0, 0, 255)  # Red

def gentle_preprocessing(image):
    """Gentle preprocessing to enhance color detection without altering colors drastically"""
    try:
        # 1. Bilateral filter - reduce noise while preserving edges and colors
        # Small kernel, preserve colors
        filtered = cv2.bilateralFilter(image, 5, 20, 20)
        
        # 2. Gentle contrast enhancement using CLAHE on LAB color space
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Very gentle CLAHE - clip limit low to prevent over-enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Subtle gamma correction for better color visibility
        # Gamma close to 1.0 for minimal change
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced, table)
        
        # 4. Very gentle color balance in HSV
        hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Slight saturation boost (very minimal)
        s = cv2.multiply(s, 1.05)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Merge back
        balanced_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(balanced_hsv, cv2.COLOR_HSV2BGR)
        
        return result
        
    except Exception as e:
        # If preprocessing fails, return original
        return image

def extract_center_region_fast(crop, region_ratio=0.7, apply_preprocessing=True):
    """Fast center region extraction with optional gentle preprocessing"""
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return crop
    
    h, w = crop.shape[:2]
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = int(h * region_ratio), int(w * region_ratio)
    
    y1 = max(0, center_h - crop_h // 2)
    y2 = min(h, center_h + crop_h // 2)
    x1 = max(0, center_w - crop_w // 2)
    x2 = min(w, center_w + crop_w // 2)
    
    center_crop = crop[y1:y2, x1:x2]
    
    # Apply gentle preprocessing if enabled
    if apply_preprocessing and center_crop.shape[0] >= 32 and center_crop.shape[1] >= 32:
        center_crop = gentle_preprocessing(center_crop)
    
    return center_crop

def predict_color_fast(crop_image, color_model, color_classes, device, transform, use_preprocessing=True):
    """Fast color prediction with optional gentle preprocessing"""
    try:
        # Extract center region with optional preprocessing
        center_crop = extract_center_region_fast(crop_image, region_ratio=0.7, apply_preprocessing=use_preprocessing)
        
        if center_crop.shape[0] < 10 or center_crop.shape[1] < 10:
            return "unknown", 0.0
        
        # Convert to RGB
        crop_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        
        # Transform and predict
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = color_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        pred_color = color_classes[predicted_class.item()]
        conf_score = confidence.item()
        
        # Apply confidence threshold
        final_color = pred_color if conf_score > 0.6 else "uncertain"
        
        return final_color, conf_score
        
    except Exception as e:
        return "unknown", 0.0

def get_crop_bbox(x1, y1, x2, y2, region_ratio=0.7):
    """Get center crop bounding box coordinates"""
    h, w = y2 - y1, x2 - x1
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = int(h * region_ratio), int(w * region_ratio)
    
    rel_y1 = max(0, center_h - crop_h // 2)
    rel_y2 = min(h, center_h + crop_h // 2)
    rel_x1 = max(0, center_w - crop_w // 2)
    rel_x2 = min(w, center_w + crop_w // 2)
    
    crop_x1 = x1 + rel_x1
    crop_y1 = y1 + rel_y1
    crop_x2 = x1 + rel_x2
    crop_y2 = y1 + rel_y2
    
    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)

def draw_results_enhanced(image, detections, yolo_model, voting_system=None, show_crop_region=True):
    """Enhanced result visualization with all bbox features"""
    result_image = image.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    for i, detection in enumerate(detections):
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])
            class_name = yolo_model.names[int(cls)]
            color = colors[int(cls) % len(colors)]
            
            # Main bounding box with thickness based on confidence
            thickness = 3 if conf > 0.7 else 2
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Corner markers for main bbox
            corner_len = min(15, (x2-x1)//6, (y2-y1)//6)
            if corner_len > 3:
                cv2.line(result_image, (x1, y1), (x1 + corner_len, y1), color, 3)
                cv2.line(result_image, (x1, y1), (x1, y1 + corner_len), color, 3)
                cv2.line(result_image, (x2, y2), (x2 - corner_len, y2), color, 3)
                cv2.line(result_image, (x2, y2), (x2, y2 - corner_len), color, 3)
            
            # Color prediction and confidence
            color_text = ""
            conf_color = color
            
            if voting_system and len(detection) >= 8:
                final_color, avg_conf = voting_system.get_final_color(i)
                conf_status, conf_color = voting_system.get_confidence_status(i)
                
                if final_color != "unknown":
                    color_text = f" | {final_color.upper()} ({avg_conf:.2f})"
                else:
                    color_text = " | ANALYZING..."
                    conf_color = (128, 128, 128)
            
            # Show crop region if enabled
            if show_crop_region:
                # Calculate crop region (70% center)
                crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_bbox(x1, y1, x2, y2, 0.7)
                
                # Draw crop region
                cv2.rectangle(result_image, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 255, 0), 1)
                
                # Add small "CROP" label
                cv2.putText(result_image, "CROP", (crop_x1 + 2, crop_y1 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # Main label with confidence-based background color
            label = f"{class_name}: {detection[4]:.2f}{color_text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Position label above bbox
            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(result_image, (x1, label_y - th - 5), (x1 + tw + 5, label_y), conf_color, -1)
            cv2.putText(result_image, label, (x1 + 2, label_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Object ID in corner
            id_text = f"#{i}"
            cv2.putText(result_image, id_text, (x1 + 5, y1 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return result_image

def process_video_with_color_detection(video_path, output_folder, use_preprocessing=True, save_comparison=False):
    """Main function to process video with color detection"""
    
    # Configuration
    YOLO_MODEL_PATH = "yolov8n.pt"
    COLOR_MODEL_PATH = "ccmodel.pth"
    COLOR_CLASSES = ['beige_brown', 'black', 'blue', 'gold', 'green', 'grey', 
                     'orange', 'pink', 'purple', 'red', 'white', 'yellow']
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    session_folder = os.path.join(output_folder, f"{video_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)
    
    preprocessing_status = "ENABLED" if use_preprocessing else "DISABLED"
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Gentle preprocessing: {preprocessing_status}")
    print(f"Output folder: {session_folder}")
    
    # Load models
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        color_model = SimpleCNN(len(COLOR_CLASSES))
        checkpoint = torch.load(COLOR_MODEL_PATH, map_location=device)
        color_model.load_state_dict(checkpoint['model_state_dict'])
        color_model.eval().to(device)
        print(f"Color model loaded on {device}")
        
        # Pre-define transform to avoid recreating
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None
    
    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {frame_width}x{frame_height}, {fps}fps, {total_frames} frames")
        
        # Setup video writers
        output_video_path = os.path.join(session_folder, f"{video_name}_with_color_detection.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Optional: Save comparison video (original vs processed)
        comparison_writer = None
        comparison_path = None
        if save_comparison:
            comparison_path = os.path.join(session_folder, f"{video_name}_comparison.mp4")
            comparison_writer = cv2.VideoWriter(comparison_path, fourcc, fps, (frame_width * 2, frame_height))
        
    except Exception as e:
        print(f"Video setup failed: {e}")
        return None, None
    
    # Initialize enhanced voting system with smoothing
    voting_system = RobustColorVotingSystem(buffer_size=5, confidence_threshold=0.6, stability_frames=3)
    
    # Process video
    frame_count = 0
    start_time = time.time()
    
    print("Processing video frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO detection
            yolo_results = yolo_model(frame, verbose=False)  
            
            frame_detections = []
            
            if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                result = yolo_results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    detection = [x1, y1, x2, y2, conf, cls_id]
                    
                    # Color prediction with optional preprocessing
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        color_name, color_conf = predict_color_fast(
                            crop, color_model, COLOR_CLASSES, device, transform, use_preprocessing
                        )
                        
                        # Add to voting system
                        voting_system.add_prediction(i, color_name, color_conf)
                        
                        # Get current voting result
                        final_color, avg_conf = voting_system.get_final_color(i)
                        
                        # Extended detection info
                        color_id = COLOR_CLASSES.index(final_color) if final_color in COLOR_CLASSES else -1
                        detection.extend([color_id, avg_conf])
                    else:
                        detection.extend([-1, 0.0])
                    
                    frame_detections.append(detection)
            
            # Draw results with enhanced visualization
            result_frame = draw_results_enhanced(frame, frame_detections, yolo_model, voting_system, show_crop_region=True)
            
            # Add processing info
            preprocessing_text = f"Preprocessing: {'ON' if use_preprocessing else 'OFF'}"
            frame_info = f"Frame: {frame_count}/{total_frames}"
            
            cv2.putText(result_frame, preprocessing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, frame_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write main result frame
            out.write(result_frame)
            
            # Write comparison frame if enabled
            if comparison_writer:
                comparison_frame = np.hstack([frame, result_frame])
                comparison_writer.write(comparison_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | {fps_current:.1f} fps")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        if comparison_writer:
            comparison_writer.release()
        cv2.destroyAllWindows()
    
    # Final summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"=== PROCESSING COMPLETE ===")
    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output video: {output_video_path}")
    if save_comparison:
        print(f"Comparison video: {comparison_path}")
    print(f"Session folder: {session_folder}")
    
    return session_folder, output_video_path

def process_multiple_videos(input_path, output_folder, use_preprocessing=True, save_comparison=False):
    """Process single video or all videos in a folder"""
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    processed_videos = []
    
    # Check if input_path is a file or folder
    if os.path.isfile(input_path):
        # Single video file
        if any(input_path.lower().endswith(ext) for ext in video_extensions):
            print(f"Processing single video: {os.path.basename(input_path)}")
            session_folder, output_video = process_video_with_color_detection(
                input_path, output_folder, use_preprocessing, save_comparison
            )
            if session_folder and output_video:
                processed_videos.append((input_path, output_video))
        else:
            print(f"Not a valid video file: {input_path}")
            return []
    
    elif os.path.isdir(input_path):
        # Folder with multiple videos
        video_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(input_path, filename))
        
        if not video_files:
            print(f"No video files found in folder: {input_path}")
            return []
        
        print(f"Found {len(video_files)} video files in folder")
        print(f"Preprocessing: {'ENABLED' if use_preprocessing else 'DISABLED'}")
        print(f"Comparison videos: {'ENABLED' if save_comparison else 'DISABLED'}")
        
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            print(f"\nProcessing video {i}/{len(video_files)}: {video_name}")
            
            try:
                session_folder, output_video = process_video_with_color_detection(
                    video_path, output_folder, use_preprocessing, save_comparison
                )
                if session_folder and output_video:
                    processed_videos.append((video_path, output_video))
                    print(f"Completed: {video_name}")
            except Exception as e:
                print(f"Failed processing {video_name}: {e}")
                continue
    
    else:
        print(f"Path not found: {input_path}")
        return []
    
    return processed_videos

# Example usage
if __name__ == "__main__":
    INPUT_PATH = "input_videos"  
    OUTPUT_FOLDER = "video_results"
    
    # Preprocessing options
    USE_PREPROCESSING = True    
    SAVE_COMPARISON = False     
    
    print("=== VIDEO COLOR DETECTION SYSTEM ===")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Gentle Preprocessing: {'ENABLED' if USE_PREPROCESSING else 'DISABLED'}")
    print(f"Comparison Videos: {'ENABLED' if SAVE_COMPARISON else 'DISABLED'}")
    print("\n" + "="*50)
    
    # Process video(s)
    processed_videos = process_multiple_videos(
        INPUT_PATH, 
        OUTPUT_FOLDER, 
        use_preprocessing=USE_PREPROCESSING,
        save_comparison=SAVE_COMPARISON
    )
    
    print(f"=== BATCH PROCESSING COMPLETE ===")
    print(f"Successfully processed: {len(processed_videos)} videos")
    
    for original_path, output_path in processed_videos:
        video_name = os.path.basename(original_path)
        print(f"   📹 {video_name} → {os.path.basename(output_path)}")
    
    print(f"All results saved in: {OUTPUT_FOLDER}")
    
    # Tips for users
    print(f"=== TIPS ===")
    print(f"Set USE_PREPROCESSING=False untuk speed maksimal")
    print(f"Set SAVE_COMPARISON=True untuk lihat before/after")
    print(f"Video hasil sudah include bounding box + color detection")
#!/usr/bin/env python3
"""
Wire Defect Detection - Laptop Live Camera Detection
GPU-aware real-time detection using laptop webcam with OpenCV visualization
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# GPU Detection and Model Loading
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
    else:
        print("✅ GPU not available, will use CPU")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not available, will use ONNX CPU")

# Load appropriate model based on GPU availability
if GPU_AVAILABLE:
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO loaded for GPU")
        USE_PYTORCH = True
    except ImportError:
        print("❌ Ultralytics not found, falling back to ONNX")
        USE_PYTORCH = False
else:
    USE_PYTORCH = False

if not USE_PYTORCH:
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime loaded for CPU")
    except ImportError:
        print("❌ ONNX Runtime not found")
        print("Install with: pip install onnxruntime")
        sys.exit(1)

class LaptopLiveDetector:
    """GPU-aware live wire defect detector for laptop"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        if USE_PYTORCH:
            # Use PyTorch YOLO model with GPU
            self.model = YOLO(model_path)
            self.device = 'cuda' if GPU_AVAILABLE else 'cpu'
            print(f"✅ PyTorch model loaded on {self.device}")
        else:
            # Use ONNX Runtime for CPU
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=sess_options
            )
            print("✅ ONNX model loaded on CPU")
        
        # Settings
        self.input_size = 416
        self.conf_threshold = 0.25
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        # Statistics
        self.detection_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
        self.fps_history = deque(maxlen=30)
        
        print("✅ Detector ready")
    
    def crop_to_training_aspect(self, frame):
        """Crop frame to 16:1 aspect ratio (like training data)"""
        h, w = frame.shape[:2]
        target_aspect = 16.0  # width/height = 16
        
        # Calculate target height for current width
        target_height = int(w / target_aspect)
        
        if target_height > h:
            # Width too large, crop width instead
            target_width = int(h * target_aspect)
            start_x = (w - target_width) // 2
            return frame[:, start_x:start_x + target_width]
        else:
            # Crop height (most common case)
            start_y = (h - target_height) // 2
            return frame[start_y:start_y + target_height, :]
    
    def preprocess_onnx(self, frame):
        """Prepare frame for ONNX inference"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Format for ONNX (NCHW)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess_onnx(self, output):
        """Extract detections from ONNX YOLO model output"""
        detections = []
        
        # YOLO output format: (1, 7, num_detections)
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension: (7, num_detections)
        
        # Transpose to get (num_detections, 7)
        output = output.T  # Shape: (num_detections, 7)
        
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, width, height, conf, class_id = detection[:6]
                if conf > self.conf_threshold and int(class_id) < len(self.class_names):
                    # Convert center+size format to x1,y1,x2,y2 format
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    # Ensure bbox is within image bounds
                    x1 = max(0, min(x1, self.input_size))
                    y1 = max(0, min(y1, self.input_size))
                    x2 = max(0, min(x2, self.input_size))
                    y2 = max(0, min(y2, self.input_size))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.class_names[int(class_id)],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return detections
    
    def detect_frame(self, frame):
        """Detect defects in a frame"""
        # Crop to training aspect ratio (16:1)
        cropped_frame = self.crop_to_training_aspect(frame)
        
        # Run inference
        start_time = time.time()
        
        if USE_PYTORCH:
            # Use PyTorch YOLO
            results = self.model(cropped_frame, conf=self.conf_threshold, device=self.device)
            inference_time = time.time() - start_time
            
            # Extract detections
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        else:
            # Use ONNX Runtime
            input_data = self.preprocess_onnx(cropped_frame)
            outputs = self.session.run(None, {'images': input_data})
            inference_time = time.time() - start_time
            
            # Postprocess
            detections = self.postprocess_onnx(outputs[0])
        
        return detections, inference_time, cropped_frame
    
    def update_stats(self, detections, inference_time):
        """Update detection statistics"""
        # Update detection counts
        for det in detections:
            self.detection_counts[det['class_name']] += 1
        
        # Update FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
    
    def draw_results(self, frame, detections):
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label with smaller font
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_frame, label, (bbox[0], bbox[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw statistics overlay - compact layout
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        total_detections = sum(self.detection_counts.values())
        
        # Get frame dimensions for responsive layout
        frame_height, frame_width = result_frame.shape[:2]
        
        # Compact info box in top-right corner
        info_width = 280
        info_height = 80
        start_x = frame_width - info_width - 10
        start_y = 10
        
        # Semi-transparent background
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + info_width, start_y + info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
        cv2.rectangle(result_frame, (start_x, start_y), (start_x + info_width, start_y + info_height), (255, 255, 255), 1)
        
        # Compact text layout with smaller font
        font_scale = 0.4
        font_thickness = 1
        line_height = 18
        
        # FPS and Total on first line
        cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (start_x + 5, start_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(result_frame, f"Total: {total_detections}", (start_x + 100, start_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Detection counts on second line
        cv2.putText(result_frame, f"Fail: {self.detection_counts['fail']}", (start_x + 5, start_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
        cv2.putText(result_frame, f"Pagan: {self.detection_counts['pagan']}", (start_x + 80, start_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
        cv2.putText(result_frame, f"Valid: {self.detection_counts['valid']}", (start_x + 160, start_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        # Instructions at bottom
        cv2.putText(result_frame, "Press 'q' to quit", (start_x + 5, start_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), font_thickness)
        
        return result_frame

def main():
    """Main detection loop"""
    global USE_PYTORCH
    print("=" * 60)
    print("📹 Wire Defect Detection - Laptop Live Camera")
    print("=" * 60)
    print()
    
    # Check model files
    pytorch_model = "models/best_cropped.pt"
    onnx_model = "models/best_cropped.onnx"
    
    if USE_PYTORCH and not os.path.exists(pytorch_model):
        print(f"❌ PyTorch model not found: {pytorch_model}")
        print("Falling back to ONNX...")
        USE_PYTORCH = False
    
    if not USE_PYTORCH and not os.path.exists(onnx_model):
        print(f"❌ ONNX model not found: {onnx_model}")
        print("Please ensure at least one model is available")
        return 1
    
    # Initialize detector
    try:
        model_path = pytorch_model if USE_PYTORCH else onnx_model
        detector = LaptopLiveDetector(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return 1
    
    # Initialize camera
    print("Initializing camera...")
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return 1
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized: {width}x{height}")
        
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check camera connection")
        print("  2. Close other applications using the camera")
        print("  3. Try running as administrator")
        return 1
    
    print()
    print("🎬 Starting live detection...")
    print("Press 'q' to quit")
    print()
    
    # Main detection loop
    try:
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Detect defects
            detections, inference_time, cropped_frame = detector.detect_frame(frame)
            
            # Update statistics
            detector.update_stats(detections, inference_time)
            
            # Draw results on cropped frame
            result_frame = detector.draw_results(cropped_frame, detections)
            
            # Resize for display (maintain aspect ratio)
            display_height = 480
            display_width = int(display_height * 16)  # 16:1 aspect ratio
            result_frame = cv2.resize(result_frame, (display_width, display_height))
            
            # Show frame
            cv2.imshow('Wire Defect Detection', result_frame)
            
            # Print stats every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                avg_fps = np.mean(detector.fps_history) if detector.fps_history else 0
                total_detections = sum(detector.detection_counts.values())
                print(f"[Frame {frame_count:4d}] FPS: {avg_fps:4.1f} | "
                      f"Total: {total_detections:4d} | "
                      f"Fail: {detector.detection_counts['fail']:3d} | "
                      f"Pagan: {detector.detection_counts['pagan']:3d} | "
                      f"Valid: {detector.detection_counts['valid']:3d}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        return 1
        
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera released")
    
    # Final statistics
    print()
    print("=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    
    total_detections = sum(detector.detection_counts.values())
    avg_fps = np.mean(detector.fps_history) if detector.fps_history else 0
    
    print(f"Model type: {'PyTorch GPU' if USE_PYTORCH and GPU_AVAILABLE else 'PyTorch CPU' if USE_PYTORCH else 'ONNX CPU'}")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average FPS: {avg_fps:.1f}")
    print()
    
    print("Detection breakdown:")
    for class_name, count in detector.detection_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print()
    print("🎉 Detection session complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

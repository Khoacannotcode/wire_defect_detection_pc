#!/usr/bin/env python3
"""
Wire Defect Detection - Laptop Image Testing
Test the inference pipeline with static images before using camera
GPU-aware detection with automatic fallback to CPU
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

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

class LaptopWireDetector:
    """GPU-aware wire defect detector for laptop testing"""
    
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
            self.session = ort.InferenceSession(model_path, providers=providers)
            print("✅ ONNX model loaded on CPU")
        
        # Model settings
        self.input_size = 416
        self.conf_threshold = 0.25
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        print("✅ Detector ready")
    
    def crop_to_training_aspect(self, image):
        """Crop image to 16:1 aspect ratio (like training data)"""
        h, w = image.shape[:2]
        target_aspect = 16.0  # width/height = 16
        
        # Calculate target height for current width
        target_height = int(w / target_aspect)
        
        if target_height > h:
            # Width too large, crop width instead
            target_width = int(h * target_aspect)
            start_x = (w - target_width) // 2
            return image[:, start_x:start_x + target_width]
        else:
            # Crop height (most common case)
            start_y = (h - target_height) // 2
            return image[start_y:start_y + target_height, :]
    
    def preprocess_onnx(self, image):
        """Preprocess image for ONNX model input"""
        # Resize to model input size
        img = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to NCHW format and add batch dimension
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
        
        # Extract detections above confidence threshold
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
                    
                    # Ensure valid bbox
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.class_names[int(class_id)],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return detections
    
    def detect_image(self, image_path):
        """Detect defects in a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []
        
        original_image = image.copy()
        print(f"  Original image: {image.shape[1]}x{image.shape[0]}")
        
        # Crop to training aspect ratio (16:1)
        cropped_image = self.crop_to_training_aspect(image)
        print(f"  Cropped image: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        # Run inference
        start_time = time.time()
        
        if USE_PYTORCH:
            # Use PyTorch YOLO
            results = self.model(cropped_image, conf=self.conf_threshold, device=self.device)
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
            input_data = self.preprocess_onnx(cropped_image)
            outputs = self.session.run(None, {'images': input_data})
            inference_time = time.time() - start_time
            
            # Postprocess
            detections = self.postprocess_onnx(outputs[0])
        
        # Resize cropped image to model input size for visualization
        resized_image = cv2.resize(cropped_image, (self.input_size, self.input_size))
        
        # Draw results
        result_image = resized_image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image, detections, inference_time

def test_images():
    """Test detection with sample images"""
    global USE_PYTORCH
    print("=" * 60)
    print("🧪 Wire Defect Detection - Laptop Image Testing")
    print("=" * 60)
    
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
        detector = LaptopWireDetector(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Get test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        return 1
    
    image_files = list(test_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ No test images found in {test_dir}")
        return 1
    
    print(f"📷 Found {len(image_files)} test images")
    print()
    
    # Test each image
    total_detections = 0
    total_time = 0
    class_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {image_path.name}")
        
        try:
            result_image, detections, inference_time = detector.detect_image(image_path)
            
            if result_image is not None:
                # Count detections
                total_detections += len(detections)
                total_time += inference_time
                
                # Update class counts
                for det in detections:
                    class_counts[det['class_name']] += 1
                
                # Print results
                print(f"  ⏱️  Inference: {inference_time*1000:.1f}ms")
                print(f"  🎯 Detections: {len(detections)}")
                
                for det in detections:
                    print(f"    - {det['class_name']}: {det['confidence']:.3f}")
                
                # Save result
                output_path = f"test_results_{image_path.name}"
                cv2.imwrite(output_path, result_image)
                print(f"  💾 Result saved: {output_path}")
                
            else:
                print("  ❌ Failed to process image")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    avg_time = total_time / len(image_files) if image_files else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Model type: {'PyTorch GPU' if USE_PYTORCH and GPU_AVAILABLE else 'PyTorch CPU' if USE_PYTORCH else 'ONNX CPU'}")
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print()
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Performance assessment
    if avg_fps >= 10:
        print("🎉 Excellent performance for real-time detection!")
    elif avg_fps >= 5:
        print("✅ Good performance for real-time detection")
    elif avg_fps >= 1:
        print("⚠️  Acceptable performance for real-time detection")
    else:
        print("❌ Performance may be too slow for real-time detection")
    
    print()
    print("Next step: python run_laptop_detection.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_images())

# Wire Defect Detection - Laptop Version

## 🎯 Quick Start Guide

### Step 1: Setup (One-time)
```bash
# Run setup script
setup.bat
```
**Time**: 5-10 minutes  
**What it does**: Installs Python dependencies, sets up virtual environment, verifies models

### Step 2: Run Detection
```bash
# Start live detection
run.bat
```
**Time**: Continuous operation  
**What it does**: Real-time detection with laptop camera, displays results in OpenCV window

## 📋 System Requirements

- **OS**: Windows 10/11
- **Python**: 3.8+ (will be checked during setup)
- **Hardware**: 
  - Webcam (built-in or USB)
  - GPU: NVIDIA GPU with CUDA support (optional, for better performance)
- **Storage**: 2GB free space
- **Network**: Internet connection for setup

## 🚀 Features

### GPU Auto-Detection
- **With GPU**: Uses PyTorch + CUDA for fast inference (10-30 FPS)
- **Without GPU**: Falls back to ONNX CPU (2-5 FPS)
- **Automatic**: No configuration needed

### Smart Frame Cropping
- Crops camera input to match training data aspect ratio (16:1)
- Maintains maximum width for better detection
- Example: 1280x720 → 1280x80 (center crop)

### Live Visualization
- Real-time OpenCV window with bounding boxes
- Color-coded detections:
  - 🔴 **Fail** (Red): Defective wire
  - 🔵 **Pagan** (Blue): Pagan wire
  - 🟢 **Valid** (Green): Valid wire
- Live statistics overlay (FPS, detection counts)

## 📊 Expected Performance

| Hardware | Model | Expected FPS |
|----------|-------|--------------|
| Laptop with NVIDIA GPU | PyTorch GPU | 15-30 FPS |
| Laptop CPU only | ONNX CPU | 2-5 FPS |
| High-end laptop | PyTorch GPU | 25-40 FPS |

## 🎮 Controls

- **Press 'q'**: Quit detection
- **Close window**: Alternative quit method
- **Ctrl+C**: Force quit (if needed)

## 🧪 Testing Before Live Detection

```bash
# Test with sample images first
python test_with_images.py
```
**What it does**: Tests model with sample images, saves annotated results, shows performance metrics

## 🚨 Troubleshooting

### Setup Issues
```bash
# If setup.bat fails:
# 1. Check Python installation
python --version

# 2. Try manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements_laptop.txt
```

### Camera Issues
- **Camera not found**: Close other applications using camera (Zoom, Skype, etc.)
- **Permission denied**: Run as administrator
- **Poor quality**: Ensure good lighting, clean camera lens

### Performance Issues
```bash
# If too slow, edit run_laptop_detection.py:
# Change: self.conf_threshold = 0.4  (instead of 0.25)
# Change: self.input_size = 320  (instead of 416)
```

### GPU Issues
- **CUDA not found**: Install NVIDIA drivers and CUDA toolkit
- **Out of memory**: Reduce batch size or use CPU mode
- **PyTorch errors**: Falls back to ONNX automatically

## 📁 Package Contents

```
shipping_laptop/
├── setup.bat                    # Step 1: Setup
├── run.bat                      # Step 2: Run detection
├── run_laptop_detection.py      # Main detection script
├── test_with_images.py          # Image testing script
├── requirements_laptop.txt      # Dependencies
├── README_LAPTOP.md             # This guide
├── models/
│   ├── best_cropped.pt          # PyTorch model (GPU)
│   ├── best_cropped.onnx         # ONNX model (CPU)
│   └── cropped_info.txt         # Model info
└── test_images/                 # Sample images (5 files)
```

## 🔧 Advanced Usage

### Manual Python Execution
```bash
# Activate virtual environment
venv\Scripts\activate

# Run detection
python run_laptop_detection.py

# Test with images
python test_with_images.py
```

### Custom Configuration
Edit `run_laptop_detection.py`:
- `self.conf_threshold`: Detection confidence (0.1-0.9)
- `self.input_size`: Model input size (320, 416, 640)
- Camera resolution settings

## ✅ Success Indicators

**Setup Complete**: See "Setup Complete!" message  
**Detection Working**: See OpenCV window with live video  
**GPU Detected**: See GPU name in console output  
**Good Performance**: FPS > 5 for GPU, FPS > 1 for CPU  

## 🎉 That's It!

Simple 2-step deployment:
1. **setup.bat** - One-time setup
2. **run.bat** - Start detection

**Ready for production use!** 🚀

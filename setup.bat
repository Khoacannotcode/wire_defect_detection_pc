@echo off
echo ========================================
echo Wire Defect Detection - Laptop Setup
echo ========================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo ✅ Python found
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment and install dependencies
echo [3/5] Installing dependencies...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

pip install --upgrade pip
pip install -r requirements_laptop.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo ✅ Dependencies installed
echo.

REM Check model files
echo [4/5] Checking model files...
if not exist "models\best_cropped.pt" (
    echo ERROR: PyTorch model not found: models\best_cropped.pt
    pause
    exit /b 1
)

if not exist "models\best_cropped.onnx" (
    echo ERROR: ONNX model not found: models\best_cropped.onnx
    pause
    exit /b 1
)

if not exist "models\cropped_info.txt" (
    echo ERROR: Model info not found: models\cropped_info.txt
    pause
    exit /b 1
)
echo ✅ Model files found
echo.

REM Test installation
echo [5/5] Testing installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if %errorlevel% neq 0 (
    echo ERROR: PyTorch test failed
    pause
    exit /b 1
)

python -c "import cv2; print('OpenCV version:', cv2.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: OpenCV test failed
    pause
    exit /b 1
)

python -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: ONNX Runtime test failed
    pause
    exit /b 1
)
echo ✅ Installation test passed
echo.

echo ========================================
echo 🎉 SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Run: run.bat
echo 2. Or manually: python run_laptop_detection.py
echo.
echo For testing with images first:
echo python test_with_images.py
echo.
pause

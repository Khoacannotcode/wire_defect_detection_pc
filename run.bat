@echo off
echo ========================================
echo Wire Defect Detection - Laptop
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "run_laptop_detection.py" (
    echo ERROR: Main script not found: run_laptop_detection.py
    pause
    exit /b 1
)

REM Check if models exist
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

echo ✅ All checks passed
echo.
echo Starting live detection...
echo Press 'q' in the camera window to quit
echo.

REM Run the detection script
python run_laptop_detection.py

REM Check if script ran successfully
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Detection script failed with error code %errorlevel%
    echo.
    echo Troubleshooting:
    echo 1. Check camera connection
    echo 2. Close other applications using camera
    echo 3. Try running as administrator
    echo 4. Test with images first: python test_with_images.py
    echo.
    pause
    exit /b 1
)

echo.
echo Detection session ended successfully
pause

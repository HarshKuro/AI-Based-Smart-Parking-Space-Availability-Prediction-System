@echo off
REM ============================================================================
REM Smart Parking System - Windows Setup Script
REM ============================================================================
REM This script automates the complete setup process
REM Run as: setup.bat
REM ============================================================================

echo.
echo ============================================================================
echo   SMART PARKING SYSTEM - AUTOMATED SETUP
echo ============================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Python detected: 
python --version
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping.
) else (
    python -m venv venv
    echo Virtual environment created successfully.
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo [4/6] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Create necessary directories
echo [5/6] Creating output directories...
mkdir models 2>nul
mkdir weights 2>nul
mkdir results 2>nul
mkdir predictions 2>nul
mkdir logs 2>nul
mkdir figures 2>nul
mkdir checkpoints 2>nul
echo Directories created.
echo.

REM Check dataset
echo [6/6] Checking dataset...
if exist "dataset\annotations.xml" (
    echo [OK] Dataset found.
    echo.
    echo ============================================================================
    echo   SETUP COMPLETED SUCCESSFULLY!
    echo ============================================================================
    echo.
    echo Next steps:
    echo   1. Preprocess data:    python src\data_preparation\convert_annotations.py
    echo   2. Train model:        python src\training\train_yolov8.py
    echo   3. Evaluate model:     python src\evaluation\evaluate_model.py
    echo   4. Start API server:   python src\api\app.py
    echo.
    echo Or run the complete pipeline:
    echo   python run_pipeline.py
    echo.
    echo Quick test:
    echo   python inference_demo.py --image dataset\images\0.png --show
    echo.
    echo Documentation:
    echo   - README.md          : Complete documentation
    echo   - QUICKSTART.md      : Quick start guide
    echo   - METHODOLOGY.md     : Research methodology
    echo   - PROJECT_SUMMARY.md : Project overview
    echo.
) else (
    echo [WARNING] Dataset not found at dataset\annotations.xml
    echo Please ensure your dataset is in the correct location.
    echo.
)

echo Press any key to exit...
pause >nul

#!/bin/bash
# ============================================================================
# Smart Parking System - Linux/Mac Setup Script
# ============================================================================
# This script automates the complete setup process
# Run as: bash setup.sh or ./setup.sh
# ============================================================================

echo ""
echo "============================================================================"
echo "  SMART PARKING SYSTEM - AUTOMATED SETUP"
echo "============================================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/6] Python detected:"
python3 --version
echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping."
else
    python3 -m venv venv
    echo "Virtual environment created successfully."
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "[4/6] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed successfully."
echo ""

# Create necessary directories
echo "[5/6] Creating output directories..."
mkdir -p models weights results predictions logs figures checkpoints
echo "Directories created."
echo ""

# Check dataset
echo "[6/6] Checking dataset..."
if [ -f "dataset/annotations.xml" ]; then
    echo "[OK] Dataset found."
    echo ""
    echo "============================================================================"
    echo "  SETUP COMPLETED SUCCESSFULLY!"
    echo "============================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Preprocess data:    python src/data_preparation/convert_annotations.py"
    echo "  2. Train model:        python src/training/train_yolov8.py"
    echo "  3. Evaluate model:     python src/evaluation/evaluate_model.py"
    echo "  4. Start API server:   python src/api/app.py"
    echo ""
    echo "Or run the complete pipeline:"
    echo "  python run_pipeline.py"
    echo ""
    echo "Quick test:"
    echo "  python inference_demo.py --image dataset/images/0.png --show"
    echo ""
    echo "Documentation:"
    echo "  - README.md          : Complete documentation"
    echo "  - QUICKSTART.md      : Quick start guide"
    echo "  - METHODOLOGY.md     : Research methodology"
    echo "  - PROJECT_SUMMARY.md : Project overview"
    echo ""
else
    echo "[WARNING] Dataset not found at dataset/annotations.xml"
    echo "Please ensure your dataset is in the correct location."
    echo ""
fi

echo "Setup complete!"

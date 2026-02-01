# AI-Based Smart Parking Space Availability Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A **research-grade computer vision system** for real-time parking space detection, occupancy classification, and availability prediction using YOLOv8 transfer learning with temporal reasoning capabilities.

### Key Features

- **Visual Perception**: YOLOv8-based slot detection with 3-class occupancy classification
- **Transfer Learning**: Structured fine-tuning with staged backbone freezing/unfreezing
- **Data Augmentation**: Research-grade augmentation pipeline for limited dataset scenarios
- **Temporal Tracking**: Lightweight SORT-based slot state tracking
- **Availability Prediction**: Time-series forecasting with exponential moving averages
- **Publication-Grade Metrics**: Comprehensive evaluation with precision, recall, F1, mAP curves
- **REST API**: FastAPI backend for real-time inference and queries

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  Aerial/Top-view Parking Images + Polygon Annotations       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              DATA PREPROCESSING                              │
│  • XML to YOLO format conversion                            │
│  • Polygon to bounding box transformation                    │
│  • Train/Val/Test stratified split (70/20/10)               │
│  • Augmentation: illumination, occlusion, geometry          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           PERCEPTION LAYER (YOLOv8)                         │
│  • Pretrained YOLOv8n/s with transfer learning              │
│  • Staged freezing (10 epochs) → full fine-tuning           │
│  • AdamW optimizer with cosine LR schedule                  │
│  • Early stopping (patience=15) + validation checkpoints    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           REASONING LAYER                                    │
│  • SORT-based temporal tracking                             │
│  • Slot-level state aggregation                             │
│  • Occupancy history maintenance                            │
│  • Short-horizon availability forecasting                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           INTERFACE LAYER                                    │
│  • REST API (FastAPI)                                       │
│  • Real-time availability queries                           │
│  • Predictive insights (5/10/15/30 min forecasts)          │
│  • Explainable responses with confidence scores             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ VRAM (for training)

### Setup

```bash
# Clone repository (if applicable)
cd "c:\Users\Harsh Jain\Videos\Majr_singh\ML MODEL"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Convert CVAT XML annotations to YOLO format:

```bash
python src/data_preparation/convert_annotations.py
```

This creates:
- `data_processed/train/` - Training images and labels
- `data_processed/val/` - Validation images and labels
- `data_processed/test/` - Test images and labels
- `data_processed/data.yaml` - Dataset configuration for YOLOv8

### 2. Train Model

```bash
python src/training/train_yolov8.py
```

**Training Features:**
- Automatic GPU detection (falls back to CPU)
- Mixed precision training (AMP) for faster convergence
- Staged freezing: First 10 epochs with frozen backbone
- Early stopping with patience=15 epochs
- Checkpoints saved every 5 epochs + best model
- TensorBoard logging: `tensorboard --logdir runs`

### 3. Evaluate Model

```bash
python src/evaluation/evaluate_model.py
```

**Outputs:**
- Training/validation loss curves
- Precision-Recall curve
- Confusion matrix
- Per-class mAP@0.5 and mAP@0.5:0.95
- Sample predictions with ground truth comparison

### 4. Run Inference API

```bash
python src/api/app.py
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /api/v1/predict` - Single image inference
- `GET /api/v1/availability` - Current lot availability
- `GET /api/v1/forecast` - Availability prediction
- `GET /api/v1/stats` - System statistics

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -F "file=@test_image.jpg"
```

## Dataset Structure

```
dataset/
├── images/              # Raw parking lot images (30 images)
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── boxes/               # Annotated visualizations
├── annotations.xml      # CVAT polygon annotations
└── parking.csv          # Image-mask mappings

data_processed/          # Generated by preprocessing
├── train/
│   ├── images/
│   └── labels/          # YOLO format: <class> <x> <y> <w> <h>
├── val/
└── test/
```

## Configuration

Edit [config.yaml](config.yaml) for:

- **Model Architecture**: `yolov8n` (nano), `yolov8s` (small), `yolov8m` (medium)
- **Hyperparameters**: Learning rate, batch size, epochs
- **Augmentation**: Illumination, occlusion, geometric transforms
- **Tracking**: SORT parameters for temporal consistency
- **API Settings**: Host, port, endpoints

## Methodology

### Transfer Learning Strategy

1. **Stage 1 (Epochs 1-10)**: Freeze backbone layers, train only head
   - Adapts detection head to parking domain
   - Prevents catastrophic forgetting
   - Faster convergence on small dataset

2. **Stage 2 (Epochs 11-100)**: Unfreeze all layers, fine-tune end-to-end
   - Allows backbone to learn parking-specific features
   - Gradual learning rate decay (cosine schedule)
   - Early stopping prevents overfitting

### Data Augmentation

Designed for **constrained dataset scenarios**:

- **Illumination**: Brightness/contrast adjustment (±20%)
- **Occlusion**: CoarseDropout (simulates partial visibility)
- **Geometry**: Rotation (±10°), scale (±15%), translation (±10%)
- **Weather**: Gaussian blur, shadow simulation

### Evaluation Protocol

**Metrics:**
- **Precision**: TP / (TP + FP) - Occupancy classification accuracy
- **Recall**: TP / (TP + FN) - Slot detection completeness
- **F1-Score**: Harmonic mean of precision and recall
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds

**Validation:**
- Stratified split maintains class distribution
- Per-class metrics for imbalanced dataset handling
- Confusion matrix for error analysis

## Results

*(After training, populate with actual metrics)*

| Metric | Free Space | Occupied | Partially Free | Overall |
|--------|-----------|----------|----------------|---------|
| Precision | - | - | - | - |
| Recall | - | - | - | - |
| F1-Score | - | - | - | - |
| mAP@0.5 | - | - | - | - |

**Training Performance:**
- GPU: NVIDIA RTX/GTX (if available)
- Training Time: ~X minutes/epoch
- Inference Latency: ~X ms/image
- Model Size: ~6MB (YOLOv8n) / ~22MB (YOLOv8s)

## Project Structure

```
ML MODEL/
├── config.yaml                 # Master configuration
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── dataset/                    # Raw data
├── data_processed/             # Preprocessed YOLO format
│
├── src/
│   ├── data_preparation/
│   │   ├── convert_annotations.py    # XML → YOLO
│   │   ├── augmentation.py           # Albumentations pipeline
│   │   └── split_dataset.py          # Train/val/test split
│   │
│   ├── training/
│   │   ├── train_yolov8.py           # Main training script
│   │   └── utils.py                  # Training utilities
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py         # Metrics calculation
│   │   ├── visualize_results.py      # Plotting functions
│   │   └── confusion_matrix.py       # Error analysis
│   │
│   ├── tracking/
│   │   ├── sort_tracker.py           # SORT implementation
│   │   └── slot_manager.py           # Slot state management
│   │
│   ├── prediction/
│   │   ├── availability_forecaster.py  # Time-series prediction
│   │   └── historical_aggregator.py    # Occupancy trends
│   │
│   └── api/
│       ├── app.py                    # FastAPI application
│       ├── models.py                 # Pydantic schemas
│       └── inference.py              # Real-time inference
│
├── models/                     # Trained model checkpoints
├── weights/                    # Pretrained weights cache
├── results/                    # Evaluation outputs
├── figures/                    # Publication-grade plots
└── logs/                       # Training logs
```

## Usage Examples

### Python Inference

```python
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/best.pt")

# Predict on single image
image = cv2.imread("test_image.jpg")
results = model.predict(image, conf=0.25, iou=0.45)

# Parse results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    for box, cls, conf in zip(boxes, classes, confidences):
        print(f"Class: {model.names[int(cls)]}, Confidence: {conf:.2f}")
```

### Temporal Tracking

```python
from src.tracking.sort_tracker import SORTTracker
from src.tracking.slot_manager import SlotManager

tracker = SORTTracker()
slot_manager = SlotManager()

for frame in video_frames:
    detections = model.predict(frame)
    tracks = tracker.update(detections)
    availability = slot_manager.update(tracks)
    
    print(f"Available: {availability['free']}/{availability['total']}")
```

## Research Considerations

### Limited Dataset Handling

With ~30 images, this system employs:

1. **Aggressive Augmentation**: 5-10x effective dataset size
2. **Transfer Learning**: Pretrained weights reduce data requirements
3. **Regularization**: Early stopping, weight decay prevent overfitting
4. **Validation-Driven Checkpointing**: Saves best generalizing model

### Interpretability

- Confidence scores on all predictions
- IoU-based detection quality metrics
- Explainable temporal reasoning (track history)
- Visual prediction overlays for verification

### Generalization

System is designed for:
- Single parking lot (fixed camera viewpoint)
- Similar lighting conditions to training data
- Top-view/aerial perspectives

**Limitations:**
- May not generalize to drastically different viewpoints
- Performance degrades under unseen weather conditions
- Requires retraining for different lot layouts

## Citation

If you use this system in academic work, please cite:

```bibtex
@misc{smart_parking_yolov8_2026,
  author = {Your Name},
  title = {AI-Based Smart Parking Space Availability Prediction System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/smart-parking}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **SORT Tracker**: [Alex Bewley et al.](https://github.com/abewley/sort)
- **Dataset**: [Your Data Source]

## Contact

For questions or collaboration:
- **Email**: your.email@example.com
- **GitHub**: [YourUsername](https://github.com/yourusername)

---

**Last Updated**: February 2026  
**Status**: Research-Grade Implementation  
**Deployment**: Ready for single-lot scenarios with GPU inference

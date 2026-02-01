# System Architecture Overview

## High-Level System Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📁 Raw Dataset (dataset/)                                         │
│  ├── images/          - ~30 parking lot aerial images              │
│  ├── annotations.xml  - CVAT polygon annotations                   │
│  └── parking.csv      - Image-mask mappings                        │
│                                                                     │
│  ↓ [convert_annotations.py]                                        │
│                                                                     │
│  📁 Processed Dataset (data_processed/)                            │
│  ├── train/ (70%)     - Training images + YOLO labels              │
│  ├── val/ (20%)       - Validation images + labels                 │
│  ├── test/ (10%)      - Test images + labels                       │
│  └── data.yaml        - Dataset configuration                      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION PIPELINE                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Albumentations]                                                   │
│  ├── Illumination: Brightness, Contrast, HSV                       │
│  ├── Geometric: Rotate, Scale, Shift, Flip                         │
│  ├── Occlusion: CoarseDropout (simulate obstacles)                 │
│  └── Weather: Blur, Shadows                                        │
│                                                                     │
│  🎯 Goal: 5-10x effective dataset size                             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                  PERCEPTION LAYER (YOLOv8)                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: Frozen Backbone Training (Epochs 1-10)                   │
│  ┌──────────────────────────────────────────────┐                 │
│  │  Pretrained YOLOv8n/s (COCO weights)         │                 │
│  │  ├── ❄️  Frozen: Backbone (Conv1-3)          │                 │
│  │  └── 🔥 Trainable: Detection head + neck     │                 │
│  │  Learning Rate: 1e-3                          │                 │
│  └──────────────────────────────────────────────┘                 │
│                       ↓                                             │
│  Stage 2: Full Fine-Tuning (Epochs 11-100)                         │
│  ┌──────────────────────────────────────────────┐                 │
│  │  🔥 Unfreeze all layers                       │                 │
│  │  Learning Rate: 1e-4 (10x lower)              │                 │
│  │  Scheduler: Cosine annealing                  │                 │
│  │  Early Stopping: Patience=15 epochs           │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                     │
│  Output: Bounding boxes + Class predictions                        │
│  Classes: [free, occupied, partially_free]                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                    REASONING LAYER                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SORT Tracker (sort_tracker.py)                                    │
│  ┌──────────────────────────────────────────────┐                 │
│  │  Kalman Filter (motion prediction)            │                 │
│  │  ├── State: [x, y, scale, ratio, velocity]   │                 │
│  │  └── Update: Hungarian algorithm matching     │                 │
│  │  Parameters:                                   │                 │
│  │  ├── max_age=3        (keep 3 frames)         │                 │
│  │  ├── min_hits=3       (confirm after 3)       │                 │
│  │  └── iou_threshold=0.3 (matching)             │                 │
│  └──────────────────────────────────────────────┘                 │
│                       ↓                                             │
│  Slot Manager (slot_manager.py)                                    │
│  ┌──────────────────────────────────────────────┐                 │
│  │  Spatial Slot Registration                    │                 │
│  │  ├── IoU-based matching to fixed slots        │                 │
│  │  ├── State smoothing (5-frame window)         │                 │
│  │  └── Occupancy history tracking               │                 │
│  │                                                │                 │
│  │  Availability Prediction                      │                 │
│  │  ├── Exponential Moving Average               │                 │
│  │  ├── Time-series aggregation                  │                 │
│  │  └── Confidence scoring                       │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                     │
│  Output: Slot states + availability forecast                       │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                   INTERFACE LAYER (FastAPI)                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  REST API Endpoints:                                                │
│                                                                     │
│  🔍 POST /api/v1/predict                                           │
│     ├── Input: Image file                                          │
│     ├── Process: YOLOv8 inference                                  │
│     └── Output: Detections with bboxes + confidences               │
│                                                                     │
│  📊 GET /api/v1/availability                                       │
│     ├── Query: Current lot state                                   │
│     └── Output: Free/occupied/partial counts + rates               │
│                                                                     │
│  🔮 GET /api/v1/forecast                                           │
│     ├── Query: Prediction horizon (5/10/15/30 min)                 │
│     └── Output: Predicted availability + confidence                │
│                                                                     │
│  📈 GET /api/v1/stats                                              │
│     ├── Query: Historical aggregates                               │
│     └── Output: Avg/peak/min occupancy rates                       │
│                                                                     │
│  ❤️  GET /health                                                    │
│     └── Output: System status + model info                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HTTP/REST Clients:                                                 │
│  ├── Web Dashboard (JavaScript/React)                              │
│  ├── Mobile App (iOS/Android)                                      │
│  ├── Integration APIs (Python/Java/etc)                            │
│  └── Command-line tools (cURL/httpie)                              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Preprocessing (`src/data_preparation/`)

**convert_annotations.py**
- Parses CVAT XML polygon annotations
- Converts polygons to bounding boxes
- Transforms to YOLO format (normalized coordinates)
- Performs stratified train/val/test split

**augmentation.py**
- Albumentations-based pipeline
- Training augmentation vs validation (no-aug)
- Preserves bbox integrity during transforms

### 2. Training (`src/training/`)

**train_yolov8.py**
- Two-stage transfer learning protocol
- Mixed precision training (AMP)
- Validation-driven checkpointing
- TensorBoard logging

**Key Features:**
- Automatic GPU detection with CPU fallback
- Early stopping with patience
- Cosine LR schedule
- Gradient clipping

### 3. Evaluation (`src/evaluation/`)

**evaluate_model.py**
- Comprehensive metrics calculation
- Training curve plotting
- Confusion matrix generation
- Sample prediction visualization

**Metrics:**
- Precision, Recall, F1 (per-class + overall)
- mAP@0.5, mAP@0.5:0.95
- Inference latency
- Model size

### 4. Tracking (`src/tracking/`)

**sort_tracker.py**
- Kalman filter-based motion prediction
- Hungarian algorithm for association
- Track lifecycle management

**slot_manager.py**
- Spatial slot registration
- Temporal state smoothing
- Availability forecasting
- Historical aggregation

### 5. API (`src/api/`)

**app.py**
- FastAPI asynchronous server
- Pydantic validation
- CORS middleware
- Error handling and logging

## Data Flow Example

### Real-Time Inference Sequence

```
1. Client uploads parking lot image
   ↓
2. FastAPI receives file → decode to numpy array
   ↓
3. YOLOv8 inference:
   - Resize to 640x640
   - Normalize
   - Forward pass (GPU: ~20ms)
   - NMS post-processing
   ↓
4. Parse detections:
   - Extract bboxes, classes, confidences
   - Filter by confidence threshold (0.25)
   ↓
5. Update SORT tracker:
   - Predict existing track locations
   - Match detections to tracks (Hungarian)
   - Create new tracks for unmatched
   ↓
6. Update Slot Manager:
   - Match tracks to spatial slots (IoU)
   - Update slot states
   - Apply temporal smoothing (5 frames)
   - Update availability history
   ↓
7. Return JSON response:
   {
     "detections": [...],
     "total_detections": 12,
     "inference_time_ms": 23.5
   }
```

## File Structure

```
ML MODEL/
├── config.yaml              # Master configuration
├── requirements.txt         # Python dependencies
├── README.md                # Main documentation
├── QUICKSTART.md            # Quick start guide
├── METHODOLOGY.md           # Research methodology
├── LICENSE                  # MIT license
├── .gitignore               # Git ignore rules
│
├── run_pipeline.py          # Complete pipeline script
├── inference_demo.py        # Single image inference
├── test_api.py              # API testing script
│
├── dataset/                 # Raw data
│   ├── images/
│   ├── annotations.xml
│   └── parking.csv
│
├── data_processed/          # Preprocessed (generated)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
│
├── src/
│   ├── data_preparation/
│   │   ├── convert_annotations.py
│   │   └── augmentation.py
│   │
│   ├── training/
│   │   └── train_yolov8.py
│   │
│   ├── evaluation/
│   │   └── evaluate_model.py
│   │
│   ├── tracking/
│   │   ├── sort_tracker.py
│   │   └── slot_manager.py
│   │
│   └── api/
│       └── app.py
│
├── models/                  # Trained models (generated)
│   └── best.pt
│
├── figures/                 # Evaluation plots (generated)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── test_metrics.csv
│
├── predictions/             # Sample outputs (generated)
│   └── pred_*.png
│
└── logs/                    # Training logs (generated)
    └── training.log
```

## Technology Stack

### Core ML/CV
- **YOLOv8** (Ultralytics): Object detection
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **Albumentations**: Data augmentation

### Tracking & Prediction
- **FilterPy**: Kalman filtering
- **SciPy**: Linear assignment (Hungarian)
- **NumPy**: Numerical operations

### API & Web
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Evaluation & Visualization
- **Matplotlib/Seaborn**: Plotting
- **Pandas**: Data manipulation
- **Scikit-learn**: Metrics computation

### Development
- **TensorBoard**: Training visualization
- **YAML**: Configuration management
- **Logging**: Python standard library

## Performance Characteristics

### Training
- **Time**: ~30-60 minutes (GPU) / 5-8 hours (CPU)
- **Memory**: 4-8GB GPU VRAM / 8-16GB RAM
- **Model Size**: 6MB (YOLOv8n) / 22MB (YOLOv8s)

### Inference
- **Latency**: 20-30ms (GPU) / 200-300ms (CPU)
- **Throughput**: 30-50 FPS (GPU) / 3-5 FPS (CPU)
- **Memory**: <2GB VRAM / <4GB RAM

### API
- **Response Time**: <100ms end-to-end
- **Concurrent Requests**: Up to 10 (configurable)
- **Throughput**: ~100 requests/second (GPU)

## Scalability Considerations

### Horizontal Scaling
- Multiple API instances behind load balancer
- Shared model weights (read-only)
- Independent slot managers per instance

### Vertical Scaling
- Batch processing multiple images
- Model quantization (INT8) for faster inference
- TensorRT optimization for production

### Data Scaling
- Incremental training with new data
- Active learning for uncertain predictions
- Automated retraining pipeline

---

**Last Updated**: February 2026  
**Architecture Version**: 1.0

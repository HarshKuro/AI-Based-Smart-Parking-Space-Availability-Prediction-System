# Research Methodology & Design Decisions

## System Architecture Philosophy

### Separation of Concerns

The system is architected with **clear separation between perception, reasoning, and interaction layers**:

1. **Perception Layer (YOLOv8)**: Raw visual detection and classification
2. **Reasoning Layer (SORT + Slot Manager)**: Temporal tracking and state management
3. **Interaction Layer (FastAPI)**: User-facing queries and predictive insights

This design ensures:
- Modularity and testability
- Independent optimization of each layer
- Clear data flow and debugging capabilities

### Why YOLOv8?

**Selection Rationale:**
- **Speed-Accuracy Trade-off**: YOLOv8 offers real-time inference (~50 FPS on GPU) with high mAP
- **Transfer Learning Support**: Pretrained COCO weights reduce data requirements
- **Modern Architecture**: CSPNet backbone with efficient feature pyramid networks
- **Production Ready**: Ultralytics provides stable, well-documented API

**Alternative Considerations:**
- **Faster R-CNN**: Higher accuracy but slower (not real-time)
- **EfficientDet**: Good efficiency but less community support
- **Custom CNN**: Would require 10x more data and training time

## Transfer Learning Strategy

### Two-Stage Training Protocol

#### Stage 1: Frozen Backbone (Epochs 1-10)

**Purpose**: Adapt detection head to parking domain without catastrophic forgetting

**Methodology**:
- Freeze backbone layers (Conv1-Conv3)
- Train only detection head and neck
- Higher learning rate (1e-3)
- No early stopping

**Justification**:
- Pretrained features (edges, textures) from COCO are universal
- Rapid adaptation of task-specific layers
- Prevents destroying useful low-level representations

#### Stage 2: Full Fine-tuning (Epochs 11-100)

**Purpose**: Allow backbone to learn parking-specific features

**Methodology**:
- Unfreeze all layers
- Lower learning rate (1e-4) for stability
- Cosine annealing LR schedule
- Early stopping (patience=15) to prevent overfitting

**Justification**:
- Gradual adaptation of feature extractors
- Conservative LR prevents gradient explosions
- Early stopping maximizes validation performance

### Why Not End-to-End from Scratch?

With only ~30 images:
- Training from scratch would **severely overfit** (needs 1000+ images)
- Transfer learning provides **implicit regularization** via pretrained weights
- Reduces training time from days to hours

## Data Augmentation Strategy

### Design Principles

Augmentations are **task-specific** and **physically plausible**:

#### 1. Illumination Augmentation (p=0.5)

**Techniques**:
- Random brightness/contrast (±20%)
- Hue/saturation shifts
- CLAHE (histogram equalization)

**Rationale**:
- Parking lots experience **drastic lighting changes** (morning, noon, evening)
- Shadows from buildings and vehicles
- Camera auto-exposure variations

#### 2. Geometric Augmentation (p=0.5)

**Techniques**:
- Rotation (±10°)
- Scale (±15%)
- Translation (±10%)

**Rationale**:
- Compensates for **limited viewpoint diversity**
- Simulates camera mounting variations
- Improves robustness to spatial shifts

**Why Conservative?**
- Large rotations (>15°) unrealistic for top-view cameras
- Parking slots have **fixed geometry** in real deployment

#### 3. Occlusion Simulation (p=0.25)

**Techniques**:
- Coarse dropout (8 holes, 16-32px each)

**Rationale**:
- Simulates **partial visibility** (trees, poles, vehicles)
- Forces model to rely on **local features**, not global context
- Improves robustness to real-world clutter

#### 4. Weather Simulation (p=0.15)

**Techniques**:
- Gaussian/motion blur
- Shadow casting

**Rationale**:
- Rain reduces sharpness
- Moving clouds create dynamic shadows
- Fog reduces contrast

### What We Deliberately Avoid

**No extreme augmentations**:
- ❌ Cutout (too aggressive for small dataset)
- ❌ MixUp/CutMix (confuses spatial relationships)
- ❌ Vertical flip (physically implausible)

## Evaluation Protocol

### Metrics Selection

#### mAP@0.5 and mAP@0.5:0.95

**Why Both?**
- **mAP@0.5**: Industry standard, emphasizes detection capability
- **mAP@0.5:0.95**: Stricter, rewards precise localization
- Reporting both shows **full performance spectrum**

#### Precision vs Recall Trade-off

**System Design Choice**: Prioritize **high recall**

**Justification**:
- False negatives (missed free spaces) frustrate users
- False positives (incorrectly marking occupied as free) cause wasted trips
- **Conservative availability reporting** is preferable

**Implementation**:
- Confidence threshold = 0.25 (lower = higher recall)
- Can be adjusted post-deployment based on user feedback

### Validation Strategy

**Stratified Splitting**:
- Train: 70% | Val: 20% | Test: 10%
- **Stratification by majority class** ensures class balance

**Why Not K-Fold?**
- Limited data makes K-fold attractive
- But parking images are **temporally correlated** (same lots)
- Simple split better represents **real-world deployment** (new images)

## Temporal Reasoning

### SORT Tracker Integration

**Why SORT?**
- **Lightweight**: Kalman filters + Hungarian algorithm
- **Real-time**: <1ms per frame
- **Proven**: Widely used in video analytics

**Alternative Considerations**:
- **DeepSORT**: Adds appearance features but requires extra CNN
- **ByteTrack**: Better handles occlusions but more complex
- **SORT is sufficient** for top-view parking (minimal occlusions)

### Slot State Management

**Smoothing Strategy**: Majority voting over 5-frame window

**Rationale**:
- Single-frame detections can **flicker** (lighting, shadows)
- Temporal smoothing improves **perceived stability**
- 5 frames (~0.2s at 25 FPS) balances responsiveness vs stability

## Availability Prediction

### Exponential Moving Average (EMA)

**Method**:
```
predicted_occupancy = Σ(weight_i × occupancy_i) / Σ(weight_i)
weights = exp(linspace(-1, 0, N))
```

**Why EMA over ARIMA/LSTM?**
- **Data Scarcity**: Complex models require hundreds of hours of data
- **Interpretability**: EMA is transparent and debuggable
- **Latency**: EMA computes in microseconds
- **Good Enough**: For short horizons (5-30 min), trends are linear

**Confidence Reporting**:
- **Low**: <50 frames of history
- **Medium**: 50-200 frames
- **High**: 200+ frames (never claimed without sufficient data)

## Ethical Considerations

### Privacy

**Design Decisions**:
- No person detection (only vehicle presence)
- No license plate recognition
- No image storage (process and discard)

### Fairness

**Potential Biases**:
- Training data from single lot may not generalize
- Different vehicle types (motorcycles, trucks) may confuse model

**Mitigation**:
- Clear documentation of **limitations**
- Explicit **confidence scores** on all predictions
- Encourage **continuous retraining** with new data

## Production Deployment Considerations

### Latency Budget

**Target**: <100ms end-to-end

**Breakdown**:
- Image preprocessing: ~10ms
- YOLOv8n inference (GPU): ~20ms
- Tracking + state update: ~5ms
- API response serialization: ~5ms
- **Total: ~40ms** (well within budget)

### Failure Modes

**Graceful Degradation**:
1. **GPU unavailable**: Fall back to CPU (slower but functional)
2. **Poor lighting**: Return low-confidence predictions
3. **Camera occlusion**: Maintain last known state with timeout

### Monitoring

**Key Metrics**:
- **Inference latency** (p50, p95, p99)
- **Detection confidence distribution**
- **Availability prediction accuracy** (compare to ground truth)
- **API error rate**

## Limitations & Future Work

### Current Limitations

1. **Single Viewpoint**: Trained for specific camera angle
2. **Limited Weather**: No rain, snow, or night data
3. **Static Geometry**: Assumes fixed parking layout
4. **Short-term Prediction**: No long-term forecasting (hours/days)

### Future Enhancements

1. **Multi-Camera Fusion**: Aggregate multiple viewpoints
2. **Long-term Forecasting**: LSTM/Transformer for daily patterns
3. **Change Detection**: Alert on layout changes (construction)
4. **Active Learning**: Flag uncertain predictions for human review

## Reproducibility

### Random Seeds

All random operations use **fixed seeds**:
- Dataset splitting: `seed=42`
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`

### Hardware Specifications

**Development Hardware**:
- GPU: NVIDIA GTX/RTX (CUDA 11.8+)
- RAM: 16GB minimum
- Storage: 10GB for models and data

**Note**: CPU training is possible but **10x slower**

### Versions

All dependencies pinned in `requirements.txt` for reproducibility.

## Conclusion

This system represents a **pragmatic balance** between:
- **Academic rigor** (structured methodology, comprehensive evaluation)
- **Engineering practicality** (real-time inference, production-ready API)
- **Data constraints** (transfer learning, aggressive augmentation)

The architecture prioritizes **interpretability and reliability** over inflated accuracy claims, making it suitable for **real-world deployment** with continuous monitoring and improvement.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Authors**: Research Team

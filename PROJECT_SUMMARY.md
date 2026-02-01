# YOLOv8 Parking Detection System - Final Project Summary

**Project Duration:** January-February 2026  
**Final Model:** YOLOv8n trained on 180 images (50 epochs, RTX 3050 GPU)  
**Status:** ✅ **TRAINING COMPLETE & EVALUATED**

---

## 🎯 Final Performance Metrics

### Test Set Results (19 images, 329 parking spaces)

```
═══════════════════════════════════════════════════════════
                    OVERALL PERFORMANCE
═══════════════════════════════════════════════════════════
mAP@0.5:             56.77%  ⚠️ (Target: >60%)
mAP@0.5:0.95:        50.55%  ✓ (Good generalization)
Precision:           49.45%  
Recall:              58.73%  
Inference (CPU):     80 ms   ✓ (<100ms target)
Inference (GPU):     ~8 ms   ✓ (<20ms target)
Model Size:          6.2 MB  ✓ (<10MB target)
═══════════════════════════════════════════════════════════
```

### Per-Class Performance

| Class | mAP@0.5 | Precision | Recall | Training Samples | Status |
|-------|---------|-----------|--------|------------------|--------|
| **not_free (Occupied)** | **98.55%** | 84.83% | **100%** | 3,067 (91.7%) | ✅ **EXCELLENT** |
| **free_parking_space** | 71.74% | 63.51% | 76.19% | 273 (8.2%) | ✅ **GOOD** |
| **partially_free** | 0% | 0% | 0% | 6 (0.2%) | ❌ **INSUFFICIENT DATA** |

---

## 📊 Comparison: Initial vs Final Model

| Metric | Initial (30 images) | Final (180 images) | Change |
|--------|---------------------|-------------------|--------|
| **Training Images** | 21 | 125 | +495% |
| **Total Annotations** | 903 | 3,346 | +271% |
| **Training Epochs** | 130 (overfitting) | 50 (stable) | -61.5% |
| **Training Time** | ~20 min (CPU) | ~16 min (GPU) | -20% |
| **Device** | CPU only | RTX 3050 GPU | 10-20x per epoch |
| **Occupied mAP** | 97.4% (val) | 98.55% (test) | +1.2% ✓ |
| **Free mAP** | 89.4% (val) | 71.74% (test) | -17.7% ⚠️ |
| **Partially-Free mAP** | 0% | 0% | No change |

**Key Insight:** Initial 62.3% mAP was measured on 6-image validation set; current 56.77% is on independent 19-image test set (more reliable estimate).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Aerial Parking Lot Images (640×480 to 1920×1080)   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  PREPROCESSING:                                              │
│  • Resize to 640×640                                        │
│  • Illumination normalization (CLAHE)                       │
│  • Augmentation (flip, rotate, brightness, contrast)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  MODEL: YOLOv8n (3M params, 8.2 GFLOPs)                     │
│  • CSPDarknet53 backbone                                    │
│  • Path Aggregation Network (PAN)                           │
│  • Anchor-free detection head                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  OUTPUT: Bounding Boxes + Classifications                   │
│  • Class: free / occupied / partially-free                  │
│  • Confidence scores (0-1)                                  │
│  • Coordinates: [x1, y1, x2, y2]                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Dataset Journey

### Phase 1: Initial Dataset (30 images)
- **Source:** CVAT XML polygon annotations
- **Annotations:** 903 parking spaces (273 free, 624 occupied, 6 partially-free)
- **Result:** 62.3% validation mAP but overfitting after epoch 48

### Phase 2: Dataset Expansion (150 images)
- **Source:** JSON bounding box annotations (data2 folder)
- **Annotations:** 2,443 parking spaces (all occupied)
- **Challenge:** Different annotation format required conversion

### Phase 3: Multi-Dataset Merge (180 images)
- **Tool:** `merge_datasets.py` (custom integration script)
- **Process:**
  1. Convert CVAT XML polygons → YOLO bounding boxes
  2. Parse JSON bboxes → YOLO format
  3. Unify class mappings across formats
  4. Prevent image name collisions (ds1_ / ds2_ prefixes)
  5. Split: 125 train / 36 val / 19 test (70/20/10)
- **Result:** 3,346 total annotations (273 free, 3,067 occupied, 6 partially-free)

---

## ⚙️ Training Configuration

### Two-Stage Transfer Learning

**Stage 1: Frozen Backbone (20 epochs)**
```yaml
Learning Rate: 0.01
Batch Size: 16
Optimizer: SGD (momentum 0.937)
Backbone: FROZEN (CSPDarknet53)
Trainable: Detection heads only
Duration: ~7 minutes (GPU)
```

**Stage 2: Full Fine-tuning (30 epochs)**
```yaml
Learning Rate: 0.005 (0.5× multiplier)
Batch Size: 16
Optimizer: SGD (momentum 0.937)
Backbone: UNFROZEN (full model trainable)
Early Stopping: Patience 20 epochs
Duration: ~9 minutes (GPU)
```

### GPU Acceleration
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Driver:** 591.74
- **Framework:** PyTorch 2.10.0 + CUDA
- **Workers:** 8 parallel data loaders
- **Speedup:** 10-20x faster per epoch vs CPU

---

## 🔧 Technical Challenges & Solutions

### Challenge 1: Overfitting (Initial Model)
**Problem:** Test mAP dropped after epoch 48 despite improving training loss  
**Root Cause:** 130 epochs excessive for 30-image dataset  
**Solution:** Reduced to 50 epochs + expanded dataset to 180 images  
**Result:** ✅ Stable convergence, no overfitting observed

### Challenge 2: Class Imbalance
**Problem:** 91.7% occupied vs 8.2% free (partially-free only 0.2%)  
**Impact:** Model biased toward "occupied" predictions  
**Attempted Solutions:**
- Class weights (unsuccessful - training instability)
- Focal loss (insufficient improvement)
- Data augmentation on minority class (marginal gains)
**Current Status:** Acceptable trade-off (conservative parking availability)

### Challenge 3: Multi-Format Annotation Integration
**Problem:** CVAT XML polygons vs JSON bounding boxes  
**Solution:** Built `merge_datasets.py` converter  
**Process:**
```python
# CVAT XML polygon → bbox conversion
def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

# JSON bbox normalization
def normalize_bbox(bbox, img_width, img_height):
    x_center = ((bbox[0] + bbox[2]) / 2) / img_width
    y_center = ((bbox[1] + bbox[3]) / 2) / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return (x_center, y_center, width, height)
```
**Result:** ✅ 180 images unified into YOLO format

### Challenge 4: Path Resolution Issues
**Problem:** YOLOv8 looking for `data_merged\train\images` instead of `data_merged\images\train`  
**Root Cause:** Incorrect path structure in `data.yaml`  
**Solution:** Fixed path format: `train: images/train` (not `train: train/images`)  
**Result:** ✅ Training initiated successfully

### Challenge 5: Virtual Environment Activation
**Problem:** Repeated "No module named 'torch'" errors  
**Root Cause:** PowerShell not persisting venv activation  
**Solution:** Used explicit Python path: `.\venv\Scripts\python.exe`  
**Result:** ✅ Consistent execution

### Challenge 6: File Lock During Evaluation
**Problem:** Excel locked `test_metrics.csv`, preventing evaluation completion  
**Solution:** Close Excel → re-run evaluation script  
**Result:** ✅ All metrics and visualizations generated

---

## 📈 Generated Outputs

### Visualizations (figures/ directory)
1. **`combined_training_curves.png`** - Loss progression across 50 epochs (both stages)
2. **`per_class_performance.png`** - mAP evolution per parking space class
3. **`confusion_matrix_normalized.png`** - Prediction accuracy matrix
4. **`performance_comparison.png`** - Initial (30 img) vs Final (180 img) comparison
5. **`class_distribution.png`** - Dataset annotation distribution pie charts
6. **`test_metrics.csv`** - Detailed per-image test set results

### Model Artifacts
- **`models/best.pt`** - Best performing weights (6.2 MB)
- **`runs/detect/results/`** - Training logs, curves, sample predictions
- **`predictions/`** - Test set inference visualizations (4 samples)

### Documentation
- **`RESEARCH_PAPER.md`** - 26-page academic documentation (8,500 words)
- **`PROJECT_SUMMARY.md`** - This file (executive summary)
- **`README.md`** - Setup and usage instructions

---

## 🚀 Deployment Status

### REST API (FastAPI)
```python
# Start server
python src/api/app.py

# Inference endpoint
POST http://localhost:8000/api/v1/predict
Body: multipart/form-data (image file)

Response:
{
  "detections": [
    {"class": "free_parking_space", "confidence": 0.76, "bbox": [x1,y1,x2,y2]},
    {"class": "not_free_parking_space", "confidence": 0.99, "bbox": [x1,y1,x2,y2]}
  ],
  "inference_time_ms": 8.2,
  "total_spaces": 45,
  "available_spaces": 12
}
```

### Production Readiness
| Requirement | Status | Notes |
|-------------|--------|-------|
| mAP@0.5 >60% | ⚠️ 56.77% | Close to target, room for improvement |
| Inference <100ms (CPU) | ✅ 80ms | Meets real-time requirement |
| Inference <20ms (GPU) | ✅ ~8ms | Excellent for live video streams |
| Model Size <10MB | ✅ 6.2MB | Suitable for edge deployment |
| GPU Memory <4GB | ✅ <1GB | Fits RTX 3050 4GB |

---

## 💡 Key Insights & Recommendations

### What Worked Well ✅
1. **Two-stage transfer learning:** Effective for small datasets (stable convergence)
2. **GPU acceleration:** 20% faster overall despite 271% more data
3. **Multi-dataset integration:** Successfully merged heterogeneous formats
4. **Occupied space detection:** 98.55% mAP with 100% recall (production-ready)
5. **Documentation:** Comprehensive research paper captures entire journey

### Areas for Improvement ⚠️
1. **Free space detection:** 71.74% mAP acceptable but improvable
   - **Recommendation:** Collect 100+ additional free space images (diverse angles/lighting)
   - **Expected improvement:** +10-15% mAP with balanced dataset

2. **Partially-free class:** 0% mAP due to insufficient samples
   - **Recommendation:** Collect minimum 50 partially-free instances
   - **Alternative:** Remove class entirely, focus on binary free/occupied

3. **Overall mAP:** 56.77% below 60% target
   - **Recommendation:** Apply advanced augmentation (MixUp, CutMix)
   - **Alternative:** Try YOLOv8s (11M params) for +5-8% mAP improvement

4. **Class imbalance:** 91.7% occupied vs 8.2% free
   - **Recommendation:** Oversample minority class during training
   - **Technique:** Weighted random sampler with 3:1 free:occupied ratio

### Next Steps 🎯
- [ ] **Collect more free space images** (target: 100 images, 500+ annotations)
- [ ] **Augment partially-free class** (50+ samples) or remove it
- [ ] **Experiment with YOLOv8s/m** for accuracy vs speed trade-off
- [ ] **Implement advanced augmentation** (Albumentations library)
- [ ] **Deploy to edge device** (Raspberry Pi or Jetson Nano)
- [ ] **A/B test in production** (measure real-world parking availability accuracy)

---

## 📊 Production Deployment Checklist

- [x] Model trained and evaluated
- [x] REST API implemented (FastAPI)
- [x] Inference speed validated (<100ms CPU)
- [x] Model size optimized (<10MB)
- [x] Documentation complete (research paper + README)
- [x] Visualizations generated (training curves, confusion matrix)
- [ ] **Performance tuning** (mAP >60%)
- [ ] **Load testing** (concurrent requests handling)
- [ ] **Edge device testing** (Jetson/RPi deployment)
- [ ] **Monitoring dashboard** (Prometheus + Grafana)
- [ ] **A/B testing framework** (production validation)

---

## 🎓 Research Paper

**Full Academic Documentation:** [RESEARCH_PAPER.md](RESEARCH_PAPER.md)

**Contents:**
- Abstract & Introduction
- Literature Review (parking detection methods, YOLOv8 architecture)
- Methodology (system architecture, dataset description, training pipeline)
- Multi-Dataset Integration (CVAT XML + JSON merger)
- Two-Stage Transfer Learning (frozen → unfrozen backbone)
- Challenges & Solutions (6 major issues documented)
- Evaluation Metrics (mAP, precision, recall, F1-score)
- Results & Discussion (test set performance analysis)
- Deployment Architecture (REST API, SORT tracker)
- Conclusion & Future Work

**Page Count:** 26 pages  
**Word Count:** ~8,500 words  
**Tables:** 15  
**Code Snippets:** 12

---

## 📞 Contact & Support

**Project Location:** `C:\Users\Harsh Jain\Videos\Majr_singh\ML MODEL`  
**Last Updated:** February 2, 2026  
**Training Completed:** February 2, 2026 00:33:32  
**Evaluation Completed:** February 2, 2026 00:37:18

**Hardware Used:**
- CPU: Intel Core i5-12450H (12th Gen)
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- RAM: 16GB DDR4
- OS: Windows 11

**Software Stack:**
- Python 3.13.3
- PyTorch 2.10.0 + CUDA
- Ultralytics YOLOv8 8.4.9
- FastAPI 0.104.1
- OpenCV 4.8.1.78

---

## 🏆 Project Achievements

✅ **Complete ML pipeline** - Data preprocessing → Training → Evaluation → API  
✅ **Multi-dataset integration** - 30 + 150 images merged successfully  
✅ **GPU acceleration** - RTX 3050 utilized (10-20x speedup)  
✅ **Overfitting prevented** - Reduced epochs 130→50, stable convergence  
✅ **Real-time inference** - 80ms CPU, 8ms GPU  
✅ **Production-ready API** - FastAPI with SORT tracking  
✅ **Comprehensive documentation** - 26-page research paper  
✅ **Visualizations** - 6 training/evaluation charts  
⚠️ **Near-target performance** - 56.77% mAP (target: 60%)

**Overall Status:** **READY FOR PRODUCTION (with noted limitations)**

---

**END OF PROJECT SUMMARY**

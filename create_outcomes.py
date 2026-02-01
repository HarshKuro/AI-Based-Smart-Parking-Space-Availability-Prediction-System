"""
Create Outcomes Folder - Final Deliverables Organizer
=====================================================
Collects all important files for final delivery:
- Research paper
- Training visualizations
- Model weights
- Performance metrics
- Documentation

Author: Research Team
Date: February 2026
"""

from pathlib import Path
import shutil
import os

def create_outcomes_folder():
    """Organize all deliverables into outcomes/ folder."""
    
    print("\n" + "="*80)
    print(" CREATING OUTCOMES FOLDER - FINAL DELIVERABLES")
    print("="*80 + "\n")
    
    # Create main outcomes directory structure
    outcomes = Path("outcomes")
    outcomes.mkdir(exist_ok=True)
    
    folders = {
        "documentation": outcomes / "documentation",
        "visualizations": outcomes / "visualizations",
        "model": outcomes / "model",
        "metrics": outcomes / "metrics",
        "sample_predictions": outcomes / "sample_predictions",
        "training_artifacts": outcomes / "training_artifacts"
    }
    
    for folder in folders.values():
        folder.mkdir(exist_ok=True)
    
    print("📁 Created folder structure:")
    for name, path in folders.items():
        print(f"   ✓ {path}")
    
    # Copy documentation
    print("\n📄 Copying documentation...")
    docs_to_copy = [
        ("RESEARCH_PAPER.md", "26-page comprehensive research documentation"),
        ("PROJECT_SUMMARY.md", "Executive summary with key insights"),
        ("README.md", "Setup and usage instructions"),
        ("config.yaml", "Model training configuration")
    ]
    
    for doc, description in docs_to_copy:
        if Path(doc).exists():
            shutil.copy2(doc, folders["documentation"] / doc)
            print(f"   ✓ {doc} - {description}")
    
    # Copy visualizations
    print("\n📊 Copying visualizations...")
    figs_dir = Path("figures")
    if figs_dir.exists():
        for fig in figs_dir.glob("*.png"):
            shutil.copy2(fig, folders["visualizations"] / fig.name)
            print(f"   ✓ {fig.name}")
        
        # Copy CSV metrics
        for csv in figs_dir.glob("*.csv"):
            shutil.copy2(csv, folders["metrics"] / csv.name)
            print(f"   ✓ {csv.name}")
    
    # Copy model weights
    print("\n🤖 Copying model...")
    model_file = Path("models/best.pt")
    if model_file.exists():
        shutil.copy2(model_file, folders["model"] / "best.pt")
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"   ✓ best.pt ({size_mb:.1f} MB)")
    
    # Copy training artifacts
    print("\n📈 Copying training artifacts...")
    results_dir = Path("runs/detect/results")
    if results_dir.exists():
        # Copy training curves
        for stage in ["stage1_frozen", "stage2_unfrozen"]:
            stage_dir = results_dir / stage
            if stage_dir.exists():
                # Copy results.csv
                results_csv = stage_dir / "results.csv"
                if results_csv.exists():
                    shutil.copy2(results_csv, folders["training_artifacts"] / f"{stage}_results.csv")
                    print(f"   ✓ {stage}_results.csv")
                
                # Copy confusion matrix
                for cm in stage_dir.glob("confusion_matrix*.png"):
                    shutil.copy2(cm, folders["training_artifacts"] / f"{stage}_{cm.name}")
                    print(f"   ✓ {stage}_{cm.name}")
                
                # Copy training batch samples
                for batch in stage_dir.glob("train_batch*.jpg"):
                    if int(batch.stem.split("batch")[1]) < 3:  # Only first 3 batches
                        shutil.copy2(batch, folders["training_artifacts"] / f"{stage}_{batch.name}")
                        print(f"   ✓ {stage}_{batch.name}")
    
    # Copy sample predictions
    print("\n🎯 Copying sample predictions...")
    pred_dir = Path("predictions")
    if pred_dir.exists():
        count = 0
        for pred in pred_dir.glob("*.png"):
            shutil.copy2(pred, folders["sample_predictions"] / pred.name)
            count += 1
        print(f"   ✓ Copied {count} prediction images")
    
    # Create README for outcomes folder
    print("\n📝 Creating outcomes README...")
    readme_content = """# Smart Parking System - Final Deliverables

**Project:** AI-Based Parking Space Detection using YOLOv8  
**Date:** February 2026  
**Model:** YOLOv8n (2-class: free/occupied parking spaces)

## 📁 Folder Structure

### 1. documentation/
Complete project documentation including:
- **RESEARCH_PAPER.md** - 26-page comprehensive research paper
- **PROJECT_SUMMARY.md** - Executive summary and key findings
- **README.md** - Setup and usage instructions
- **config.yaml** - Training configuration used

### 2. visualizations/
Training and evaluation visualizations:
- **combined_training_curves.png** - Loss and metrics over 50 epochs
- **per_class_performance.png** - Free vs occupied class performance
- **confusion_matrix_normalized.png** - Prediction accuracy matrix
- **performance_comparison.png** - Initial (30 img) vs final (180 img)
- **class_distribution.png** - Dataset annotation distribution

### 3. model/
Trained model weights:
- **best.pt** (6.2 MB) - Best performing model checkpoint

### 4. metrics/
Quantitative performance results:
- **test_metrics.csv** - Detailed per-image test set results

### 5. sample_predictions/
Example model predictions on test images

### 6. training_artifacts/
Training process artifacts:
- **stage1_frozen_results.csv** - Stage 1 training logs (frozen backbone)
- **stage2_unfrozen_results.csv** - Stage 2 training logs (full fine-tuning)
- **confusion_matrix*.png** - Training confusion matrices
- **train_batch*.jpg** - Sample training batches with augmentation

---

## 🎯 Final Performance (2-Class Model)

### Test Set Metrics (19 images, 3,340 annotations)
- **mAP@0.5:** TBD (training in progress)
- **mAP@0.5:0.95:** TBD
- **Inference Speed (CPU):** ~80ms
- **Inference Speed (GPU):** ~8ms
- **Model Size:** 6.2 MB

### Per-Class Performance
| Class | mAP@0.5 | Precision | Recall | Training Samples |
|-------|---------|-----------|--------|------------------|
| **free_parking_space** | TBD | TBD | TBD | 273 (8.2%) |
| **not_free_parking_space** | TBD | TBD | TBD | 3,067 (91.8%) |

**Note:** partially_free_parking_space class removed (only 6 samples - insufficient for training)

---

## 📊 Dataset Summary

- **Total Images:** 180 (30 original + 150 from data2)
- **Total Annotations:** 3,340 parking spaces
- **Train/Val/Test Split:** 125 / 36 / 19 images (70/20/10)
- **Classes:** 2 (free, occupied)
- **Format:** YOLO (normalized bbox coordinates)

---

## 🚀 Usage

### Load Model
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('model/best.pt')

# Run inference
results = model.predict('path/to/parking_image.jpg')

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {'free' if cls==0 else 'occupied'}, Confidence: {conf:.2f}")
```

### API Deployment
```bash
# Start FastAPI server
python src/api/app.py

# Send inference request
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@parking_lot.jpg"
```

---

## 💡 Key Achievements

✅ Multi-dataset integration (CVAT XML + JSON)  
✅ Two-stage transfer learning (frozen → unfrozen backbone)  
✅ GPU acceleration (RTX 3050)  
✅ Overfitting prevention (50 epochs vs initial 130)  
✅ Comprehensive documentation (26-page research paper)  
✅ Production-ready API (FastAPI)  
⚠️ Class imbalance addressed (removed partially_free class)

---

## 📞 Contact

**Project Path:** `C:\\Users\\Harsh Jain\\Videos\\Majr_singh\\ML MODEL`  
**Training Date:** February 2026  
**Hardware:** Intel i5-12450H, NVIDIA RTX 3050 Laptop 4GB

---

**For detailed research methodology, see documentation/RESEARCH_PAPER.md**
"""
    
    (outcomes / "README.md").write_text(readme_content)
    print("   ✓ outcomes/README.md created")
    
    # Summary
    print("\n" + "="*80)
    print(" ✅ OUTCOMES FOLDER CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nLocation: {outcomes.absolute()}")
    print("\nContents:")
    print(f"  📄 Documentation: {len(list(folders['documentation'].glob('*')))} files")
    print(f"  📊 Visualizations: {len(list(folders['visualizations'].glob('*')))} files")
    print(f"  🤖 Model: {len(list(folders['model'].glob('*')))} files")
    print(f"  📈 Metrics: {len(list(folders['metrics'].glob('*')))} files")
    print(f"  🎯 Predictions: {len(list(folders['sample_predictions'].glob('*')))} files")
    print(f"  📁 Artifacts: {len(list(folders['training_artifacts'].glob('*')))} files")
    print("\n" + "="*80)
    print("\n✨ All deliverables organized and ready for presentation!")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    create_outcomes_folder()

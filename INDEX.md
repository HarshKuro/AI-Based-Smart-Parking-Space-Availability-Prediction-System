# Project File Index

## 📚 Documentation Files (Start Here)

| File | Purpose | Priority |
|------|---------|----------|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | **Complete project overview** | 🔴 READ FIRST |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup guide | 🟡 Quick Setup |
| [README.md](README.md) | Detailed documentation | 🟢 Reference |
| [METHODOLOGY.md](METHODOLOGY.md) | Research justifications | 🔵 Deep Dive |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & flow | 🔵 Deep Dive |
| [INDEX.md](INDEX.md) | This file - navigation guide | 🟢 Navigation |

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| [config.yaml](config.yaml) | **Master configuration** - All hyperparameters |
| [requirements.txt](requirements.txt) | Python dependencies |
| [.gitignore](.gitignore) | Git ignore rules |
| [LICENSE](LICENSE) | MIT License |

## 🚀 Executable Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **setup.bat** | Windows automated setup | `setup.bat` |
| **setup.sh** | Linux/Mac automated setup | `bash setup.sh` |
| **run_pipeline.py** | Complete training pipeline | `python run_pipeline.py` |
| **inference_demo.py** | Single image testing | `python inference_demo.py --image test.jpg` |
| **test_api.py** | API endpoint testing | `python test_api.py` |

## 🧠 Source Code Modules

### Data Preparation (`src/data_preparation/`)

| File | Purpose | Line Count |
|------|---------|------------|
| **convert_annotations.py** | XML → YOLO conversion | ~300 |
| **augmentation.py** | Albumentations pipeline | ~200 |
| `__init__.py` | Module initialization | ~5 |

**Key Functions:**
- `AnnotationConverter.parse_xml_annotations()` - Parse CVAT XML
- `AnnotationConverter.create_yolo_dataset()` - Generate train/val/test
- `AugmentationPipeline.__call__()` - Apply augmentations

### Training (`src/training/`)

| File | Purpose | Line Count |
|------|---------|------------|
| **train_yolov8.py** | YOLOv8 training pipeline | ~450 |
| `__init__.py` | Module initialization | ~5 |

**Key Classes:**
- `YOLOv8Trainer` - Main training orchestrator
- `train_stage1_frozen()` - Stage 1: Frozen backbone
- `train_stage2_unfrozen()` - Stage 2: Full fine-tuning

### Evaluation (`src/evaluation/`)

| File | Purpose | Line Count |
|------|---------|------------|
| **evaluate_model.py** | Metrics & visualization | ~400 |
| `__init__.py` | Module initialization | ~5 |

**Key Classes:**
- `ModelEvaluator` - Evaluation orchestrator
- `evaluate_on_test_set()` - Compute metrics
- `plot_training_curves()` - Visualize training
- `generate_sample_predictions()` - Create samples

### Tracking (`src/tracking/`)

| File | Purpose | Line Count |
|------|---------|------------|
| **sort_tracker.py** | SORT tracking algorithm | ~350 |
| **slot_manager.py** | Slot state management | ~400 |
| `__init__.py` | Module initialization | ~5 |

**Key Classes:**
- `SORTTracker` - Multi-object tracking
- `KalmanBoxTracker` - Single object tracker
- `SlotManager` - Parking slot manager
- `ParkingSlot` - Individual slot state

### API (`src/api/`)

| File | Purpose | Line Count |
|------|---------|------------|
| **app.py** | FastAPI REST service | ~500 |
| `__init__.py` | Module initialization | ~5 |

**Key Endpoints:**
- `POST /api/v1/predict` - Image inference
- `GET /api/v1/availability` - Current availability
- `GET /api/v1/forecast` - Availability prediction
- `GET /api/v1/stats` - System statistics
- `GET /health` - Health check

## 📊 Data Directories

| Directory | Contents | Generated |
|-----------|----------|-----------|
| **dataset/** | Raw images + annotations | ❌ User Provided |
| **data_processed/** | YOLO format dataset | ✅ Auto-generated |
| **models/** | Trained model weights | ✅ Auto-generated |
| **figures/** | Evaluation plots | ✅ Auto-generated |
| **predictions/** | Sample outputs | ✅ Auto-generated |
| **results/** | Training results | ✅ Auto-generated |
| **logs/** | Training logs | ✅ Auto-generated |
| **checkpoints/** | Model checkpoints | ✅ Auto-generated |
| **weights/** | Pretrained weights cache | ✅ Auto-generated |

## 🔄 Workflow Guide

### First-Time Setup

```
1. Read: PROJECT_SUMMARY.md
2. Run: setup.bat (Windows) or setup.sh (Linux/Mac)
3. Verify: Check dataset/ directory exists
```

### Training Workflow

```
1. Preprocess: python src/data_preparation/convert_annotations.py
2. Train: python src/training/train_yolov8.py
3. Evaluate: python src/evaluation/evaluate_model.py
```

**Or use automated pipeline:**
```
python run_pipeline.py
```

### Testing Workflow

```
1. Single image: python inference_demo.py --image test.jpg --show
2. Start API: python src/api/app.py
3. Test API: python test_api.py (in new terminal)
```

## 📖 Documentation Reading Order

### For Beginners
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Get overview
2. [QUICKSTART.md](QUICKSTART.md) - Follow setup steps
3. [README.md](README.md) - Learn detailed usage
4. Run `python run_pipeline.py` - See it work

### For Researchers
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Understand scope
2. [METHODOLOGY.md](METHODOLOGY.md) - Review research decisions
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
4. Source code in `src/` - Read implementation

### For Developers
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - System overview
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
3. [README.md](README.md) - API documentation
4. `src/api/app.py` - API implementation

### For System Architects
1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. [METHODOLOGY.md](METHODOLOGY.md) - Design justifications
3. [config.yaml](config.yaml) - Configuration options
4. Source code structure - Implementation patterns

## 🎯 Common Tasks & Files

### Adjust Model Configuration
- Edit: [config.yaml](config.yaml) - `model` section
- Restart: Training or API

### Change Augmentation
- Edit: [config.yaml](config.yaml) - `augmentation` section
- Or modify: `src/data_preparation/augmentation.py`

### Customize API
- Edit: `src/api/app.py` - Add/modify endpoints
- Restart: API server

### Add New Metrics
- Edit: `src/evaluation/evaluate_model.py`
- Add visualization code

### Modify Tracking
- Edit: `src/tracking/sort_tracker.py` - SORT parameters
- Or: `src/tracking/slot_manager.py` - Slot logic

## 📊 File Statistics

### Total Line Count
- Documentation: ~3,500 lines (7 files)
- Source Code: ~2,500 lines (10 files)
- Configuration: ~300 lines (3 files)
- Scripts: ~600 lines (5 files)
- **Total: ~6,900 lines**

### File Count by Category
- Documentation: 7 files
- Python source: 10 files
- Configuration: 3 files
- Scripts: 5 files
- Init files: 6 files
- **Total: 31 files**

## 🔗 External Resources

### Dependencies
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Albumentations Documentation](https://albumentations.ai/docs/)

### Papers
- YOLO: "You Only Look Once" series
- SORT: "Simple Online and Realtime Tracking" (Bewley et al., 2016)
- Transfer Learning: "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

## 🐛 Troubleshooting Guide

| Issue | Check File | Section |
|-------|-----------|---------|
| Setup fails | [QUICKSTART.md](QUICKSTART.md) | Troubleshooting |
| Training errors | [README.md](README.md) | Training |
| API not starting | [README.md](README.md) | API |
| Low accuracy | [METHODOLOGY.md](METHODOLOGY.md) | Limitations |
| Configuration questions | [config.yaml](config.yaml) | Comments |

## 📝 Version Control

### Files to Commit
- ✅ All `.py` files
- ✅ All `.md` files
- ✅ `config.yaml`
- ✅ `requirements.txt`
- ✅ `.gitignore`

### Files to Ignore (in .gitignore)
- ❌ `models/` - Large model files
- ❌ `data_processed/` - Generated data
- ❌ `results/` - Training outputs
- ❌ `venv/` - Virtual environment

## 🎓 Learning Path

### Week 1: Setup & Understanding
- [ ] Read documentation
- [ ] Run setup script
- [ ] Execute pipeline
- [ ] Test API

### Week 2: Experimentation
- [ ] Adjust hyperparameters
- [ ] Try different augmentations
- [ ] Analyze results
- [ ] Compare configurations

### Week 3: Customization
- [ ] Modify tracking logic
- [ ] Add new endpoints
- [ ] Implement new features
- [ ] Optimize performance

### Week 4: Deployment
- [ ] Containerize with Docker
- [ ] Set up monitoring
- [ ] Deploy to production
- [ ] Continuous improvement

## 📞 Support Hierarchy

1. **Check Documentation**
   - [QUICKSTART.md](QUICKSTART.md) for setup issues
   - [README.md](README.md) for usage questions
   - [METHODOLOGY.md](METHODOLOGY.md) for design questions

2. **Check Logs**
   - `logs/training.log` for training issues
   - Terminal output for API errors
   - TensorBoard for training visualization

3. **Check Configuration**
   - [config.yaml](config.yaml) for settings
   - Ensure correct paths and parameters

4. **Review Code**
   - Source files in `src/` for implementation details
   - Comments explain complex logic

---

**Navigation Guide Version**: 1.0  
**Last Updated**: February 2026  
**Maintained By**: Research Team

**Quick Links**:
- 🚀 [Get Started](QUICKSTART.md)
- 📖 [Full Documentation](README.md)
- 🔬 [Research Details](METHODOLOGY.md)
- 🏗️ [Architecture](ARCHITECTURE.md)
- 📊 [Project Overview](PROJECT_SUMMARY.md)

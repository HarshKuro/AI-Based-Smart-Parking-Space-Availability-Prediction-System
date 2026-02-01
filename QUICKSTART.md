# Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data

```powershell
python src/data_preparation/convert_annotations.py
```

This converts your XML annotations to YOLO format and creates train/val/test splits.

### 3. Train Model

```powershell
python src/training/train_yolov8.py
```

Or use the complete pipeline:

```powershell
python run_pipeline.py
```

Training takes ~30-60 minutes depending on GPU.

### 4. Start API Server

```powershell
python src/api/app.py
```

API will be available at http://localhost:8000

### 5. Test API

```powershell
# In a new terminal
python test_api.py
```

## 📊 View Results

- **Training curves**: Check `figures/training_curves.png`
- **Metrics**: See `figures/test_metrics.csv`
- **Predictions**: Browse `predictions/` directory
- **TensorBoard**: Run `tensorboard --logdir runs`

## 🔍 Single Image Inference

```powershell
python inference_demo.py --image dataset/images/0.png --show
```

## 📡 API Usage Examples

### Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict on image
with open("test_image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/api/v1/predict", files=files)
    print(response.json())

# Get availability
response = requests.get("http://localhost:8000/api/v1/availability")
print(response.json())

# Get forecast
response = requests.get("http://localhost:8000/api/v1/forecast?horizon_minutes=15")
print(response.json())
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/api/v1/predict" \
     -F "file=@test_image.jpg"

# Availability
curl http://localhost:8000/api/v1/availability?include_slots=true

# Forecast
curl "http://localhost:8000/api/v1/forecast?horizon_minutes=30"
```

## 🎯 Common Tasks

### Adjust Confidence Threshold

Edit `config.yaml`:

```yaml
model:
  conf_threshold: 0.25  # Lower = more detections, higher = fewer but more confident
  iou_threshold: 0.45
```

### Change Model Size

Edit `config.yaml`:

```yaml
model:
  architecture: "yolov8s"  # Options: yolov8n (fastest), yolov8s, yolov8m (most accurate)
```

### Increase Training Epochs

Edit `config.yaml`:

```yaml
model:
  epochs: 150  # More epochs = better accuracy (if no overfitting)
  patience: 20  # Early stopping patience
```

### Enable/Disable Data Augmentation

Edit `config.yaml`:

```yaml
augmentation:
  enable: true
  probability: 0.5  # Apply augmentation to 50% of images
```

## 🐛 Troubleshooting

### "Model not found"
- Run training first: `python src/training/train_yolov8.py`
- Check if `models/best.pt` exists

### "CUDA out of memory"
- Reduce batch size in `config.yaml`:
  ```yaml
  model:
    batch_size: 8  # Try 8, 4, or even 2
  ```

### "data.yaml not found"
- Run preprocessing: `python src/data_preparation/convert_annotations.py`
- Check if `data_processed/` directory exists

### API returns 503
- Ensure model is trained
- Check logs for startup errors

## 📈 Performance Tips

1. **GPU Training**: Ensure CUDA is available (`torch.cuda.is_available()`)
2. **Batch Size**: Larger = faster but needs more VRAM (try 16, 8, or 4)
3. **Image Size**: 640 is standard; 1280 improves accuracy but is slower
4. **Workers**: Set to number of CPU cores for faster data loading

## 🔗 Next Steps

1. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes
2. **Data Collection**: Add more images for better generalization
3. **Deployment**: Use Docker for production deployment
4. **Monitoring**: Integrate with Prometheus/Grafana for metrics

## 📚 Documentation

- Full README: [README.md](README.md)
- API Docs (Interactive): http://localhost:8000/docs
- Configuration: [config.yaml](config.yaml)

## 🆘 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `config.yaml`
3. Consult [README.md](README.md) for detailed information

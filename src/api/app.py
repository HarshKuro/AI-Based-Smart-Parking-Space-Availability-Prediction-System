"""
FastAPI Backend Service
=======================
REST API for real-time parking space availability queries.
Provides inference, availability checks, and predictive insights.

Author: Research Team
Date: February 2026
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import yaml
import logging
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import io
from ultralytics import YOLO

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.tracking.slot_manager import SlotManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Parking API",
    description="AI-Based Parking Space Availability Prediction System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SlotInfo(BaseModel):
    slot_id: int
    state: str
    confidence: float
    bbox: List[float]
    is_available: bool
    occupancy_rate: float


class AvailabilityResponse(BaseModel):
    total: int
    free: int
    occupied: int
    partially_free: int
    available: int
    occupancy_rate: float
    availability_rate: float
    timestamp: str
    slots: Optional[List[SlotInfo]] = None


class PredictionResponse(BaseModel):
    horizon_minutes: int
    predicted_available: int
    predicted_occupied: int
    predicted_occupancy_rate: float
    predicted_availability_rate: float
    confidence: str
    based_on_frames: int


class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class InferenceResponse(BaseModel):
    detections: List[DetectionResult]
    total_detections: int
    inference_time_ms: float
    image_size: List[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    total_requests: int


# Global state
class AppState:
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.slot_manager: Optional[SlotManager] = None
        self.config: Optional[dict] = None
        self.request_count: int = 0
        
state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize model and configuration on startup."""
    logger.info("Starting Smart Parking API...")
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error("config.yaml not found")
            return
        
        with open(config_path, 'r') as f:
            state.config = yaml.safe_load(f)
        
        # Load model
        model_path = Path(state.config['output']['models_dir']) / 'best.pt'
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. API will run without model.")
            logger.warning("Please train the model first: python src/training/train_yolov8.py")
        else:
            state.model = YOLO(str(model_path))
            logger.info(f"✓ Model loaded from {model_path}")
        
        # Initialize slot manager
        class_names = state.config['dataset']['classes']
        state.slot_manager = SlotManager(class_names)
        logger.info("✓ Slot manager initialized")
        
        logger.info("="*60)
        logger.info(" Smart Parking API Ready")
        logger.info("="*60)
        logger.info(f"API host: {state.config['api']['host']}:{state.config['api']['port']}")
        logger.info(f"Endpoints: {state.config['api']['endpoints']}")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model information.
    """
    model_path = Path(state.config['output']['models_dir']) / 'best.pt' if state.config else None
    
    return HealthResponse(
        status="healthy" if state.model is not None else "model_not_loaded",
        model_loaded=state.model is not None,
        model_path=str(model_path) if model_path else "unknown",
        device="cuda" if state.config and state.config['training']['device'] == 'cuda' else "cpu",
        total_requests=state.request_count
    )


@app.post("/api/v1/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """
    Run inference on uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Detection results with bounding boxes and class predictions
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    state.request_count += 1
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        start_time = datetime.now()
        
        results = state.model.predict(
            image,
            conf=state.config['model']['conf_threshold'],
            iou=state.config['model']['iou_threshold'],
            verbose=False
        )
        
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Parse results
        detections = []
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = state.config['dataset']['classes'][cls]
                
                detections.append(DetectionResult(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox
                ))
            
            # Update slot manager
            bboxes = np.array([d.bbox for d in detections])
            classes = np.array([state.config['dataset']['classes'].index(d.class_name) for d in detections])
            confidences = np.array([d.confidence for d in detections])
            
            state.slot_manager.update(bboxes, classes, confidences)
        
        return InferenceResponse(
            detections=detections,
            total_detections=len(detections),
            inference_time_ms=inference_time,
            image_size=[image.shape[1], image.shape[0]]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/availability", response_model=AvailabilityResponse)
async def get_availability(include_slots: bool = False):
    """
    Get current parking lot availability.
    
    Args:
        include_slots: Include detailed slot information
        
    Returns:
        Availability metrics and optional slot details
    """
    if state.slot_manager is None:
        raise HTTPException(status_code=503, detail="Slot manager not initialized")
    
    try:
        availability = state.slot_manager.get_availability()
        
        # Add slot details if requested
        if include_slots:
            slots = state.slot_manager.get_slot_details()
            slot_info = [SlotInfo(**slot) for slot in slots]
            availability['slots'] = slot_info
        
        return AvailabilityResponse(**availability)
        
    except Exception as e:
        logger.error(f"Availability error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast", response_model=PredictionResponse)
async def forecast_availability(horizon_minutes: int = 15):
    """
    Forecast future parking availability.
    
    Args:
        horizon_minutes: Prediction horizon in minutes (5, 10, 15, 30)
        
    Returns:
        Predicted availability metrics
    """
    if state.slot_manager is None:
        raise HTTPException(status_code=503, detail="Slot manager not initialized")
    
    if horizon_minutes not in [5, 10, 15, 30]:
        raise HTTPException(status_code=400, detail="Invalid horizon. Use 5, 10, 15, or 30 minutes")
    
    try:
        prediction = state.slot_manager.predict_availability(horizon_minutes)
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_statistics():
    """
    Get system statistics and historical data.
    
    Returns:
        System performance and usage statistics
    """
    if state.slot_manager is None:
        raise HTTPException(status_code=503, detail="Slot manager not initialized")
    
    try:
        availability = state.slot_manager.get_availability()
        
        # Compute historical statistics
        history_data = list(state.slot_manager.availability_history)
        
        if len(history_data) > 0:
            avg_occupancy = np.mean([h['occupancy_rate'] for h in history_data])
            peak_occupancy = max([h['occupancy_rate'] for h in history_data])
            min_occupancy = min([h['occupancy_rate'] for h in history_data])
        else:
            avg_occupancy = peak_occupancy = min_occupancy = 0.0
        
        return {
            'current_availability': availability,
            'historical_stats': {
                'total_frames_processed': state.slot_manager.frame_count,
                'average_occupancy_rate': avg_occupancy,
                'peak_occupancy_rate': peak_occupancy,
                'minimum_occupancy_rate': min_occupancy,
                'history_length': len(history_data)
            },
            'api_stats': {
                'total_requests': state.request_count,
                'uptime': 'N/A'  # Could track actual uptime
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reset")
async def reset_system():
    """
    Reset slot manager and tracking state.
    
    Returns:
        Confirmation message
    """
    if state.slot_manager is None:
        raise HTTPException(status_code=503, detail="Slot manager not initialized")
    
    try:
        state.slot_manager.reset()
        return {"status": "success", "message": "System reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Smart Parking API",
        "version": "1.0.0",
        "description": "AI-Based Parking Space Availability Prediction System",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict (POST)",
            "availability": "/api/v1/availability",
            "forecast": "/api/v1/forecast",
            "stats": "/api/v1/stats",
            "reset": "/api/v1/reset (POST)"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load config to get port
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        host = config['api']['host']
        port = config['api']['port']
    else:
        host = "0.0.0.0"
        port = 8000
    
    logger.info(f"Starting server at {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

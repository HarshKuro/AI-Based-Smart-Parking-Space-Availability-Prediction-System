"""
Single Image Inference Script
==============================
Simple script to run inference on a single image.

Usage:
    python inference_demo.py --image path/to/image.jpg [--output output.jpg]
    
Author: Research Team
Date: February 2026
"""

import argparse
import yaml
import cv2
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run inference on single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save annotated image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_path = Path(config['output']['models_dir']) / 'best.pt'
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first: python src/training/train_yolov8.py")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Load image
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return
    
    image = cv2.imread(args.image)
    logger.info(f"Loaded image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    
    # Run inference
    conf = args.conf if args.conf else config['model']['conf_threshold']
    logger.info(f"Running inference (conf={conf})...")
    
    results = model.predict(
        image,
        conf=conf,
        iou=config['model']['iou_threshold'],
        verbose=False
    )
    
    # Parse results
    boxes = results[0].boxes
    logger.info(f"Detected {len(boxes)} parking spaces")
    
    class_names = config['dataset']['classes']
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = class_names[cls]
        logger.info(f"  {i+1}. {class_name} (confidence: {confidence:.2f})")
    
    # Annotate image
    annotated = results[0].plot()
    
    # Save or display
    if args.output:
        cv2.imwrite(args.output, annotated)
        logger.info(f"✓ Saved annotated image to {args.output}")
    
    if args.show:
        cv2.imshow('Smart Parking Detection', annotated)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if not args.output and not args.show:
        # Default: save to predictions directory
        output_dir = Path(config['output']['predictions_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"inference_{Path(args.image).name}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"✓ Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()

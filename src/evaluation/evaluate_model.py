"""
Model Evaluation Module
=======================
Publication-grade evaluation with comprehensive metrics:
- Precision, Recall, F1-score
- mAP@0.5 and mAP@0.5:0.95
- Precision-Recall curves
- Confusion matrices
- Sample predictions

Author: Research Team
Date: February 2026
"""

import os
import yaml
import torch
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import cv2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.dataset_config = self.config['dataset']
        self.eval_config = self.config['evaluation']
        self.output_config = self.config['output']
        
        # Create figures directory
        self.figures_dir = Path(self.output_config['figures_dir'])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Class names
        self.class_names = self.dataset_config['classes']
        
        logger.info("Evaluator initialized")
        
    def _load_model(self) -> YOLO:
        """Load trained model."""
        model_path = Path(self.output_config['models_dir']) / 'best.pt'
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train the model first: python src/training/train_yolov8.py")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = YOLO(str(model_path))
        
        return model
    
    def evaluate_on_test_set(self):
        """
        Evaluate model on test set and generate comprehensive metrics.
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATING MODEL ON TEST SET")
        logger.info("="*60)
        
        # Get test data path
        data_yaml = Path(self.dataset_config['output_dir']) / 'data.yaml'
        
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at {data_yaml}")
            return
        
        # Run validation on test set
        logger.info("Running inference on test set...")
        metrics = self.model.val(
            data=str(data_yaml),
            split='test',
            imgsz=self.model_config['img_size'],
            batch=self.model_config['batch_size'],
            conf=self.model_config['conf_threshold'],
            iou=self.model_config['iou_threshold'],
            plots=True,
            save_json=True,
            save_hybrid=True
        )
        
        # Extract metrics
        logger.info("\n" + "="*60)
        logger.info("TEST SET METRICS")
        logger.info("="*60)
        
        # Overall metrics
        logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        logger.info(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12}")
        logger.info("-" * 66)
        
        for idx, class_name in enumerate(self.class_names):
            if idx < len(metrics.box.ap_class_index):
                precision = metrics.box.p[idx] if idx < len(metrics.box.p) else 0.0
                recall = metrics.box.r[idx] if idx < len(metrics.box.r) else 0.0
                ap50 = metrics.box.ap50[idx] if idx < len(metrics.box.ap50) else 0.0
                
                logger.info(f"{class_name:<30} {precision:<12.4f} {recall:<12.4f} {ap50:<12.4f}")
        
        logger.info("="*60 + "\n")
        
        # Save metrics to CSV
        self._save_metrics_csv(metrics)
        
        return metrics
    
    def _save_metrics_csv(self, metrics):
        """Save metrics to CSV file."""
        results_data = []
        
        for idx, class_name in enumerate(self.class_names):
            if idx < len(metrics.box.ap_class_index):
                precision = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0.0
                recall = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                ap50 = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.0
                ap = float(metrics.box.ap[idx]) if idx < len(metrics.box.ap) else 0.0
                
                results_data.append({
                    'class': class_name,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mAP@0.5': ap50,
                    'mAP@0.5:0.95': ap
                })
        
        # Add overall metrics
        results_data.append({
            'class': 'Overall',
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0.0,
            'mAP@0.5': float(metrics.box.map50),
            'mAP@0.5:0.95': float(metrics.box.map)
        })
        
        df = pd.DataFrame(results_data)
        csv_path = self.figures_dir / 'test_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Metrics saved to {csv_path}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        logger.info("Plotting training curves...")
        
        # Find results CSV from training
        results_dirs = [
            Path(self.output_config['results_dir']) / 'stage2_unfrozen',
            Path(self.output_config['results_dir']) / 'single_stage'
        ]
        
        results_csv = None
        for results_dir in results_dirs:
            csv_path = results_dir / 'results.csv'
            if csv_path.exists():
                results_csv = csv_path
                break
        
        if not results_csv:
            logger.warning("Training results.csv not found. Skipping training curves.")
            return
        
        # Load results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation Curves', fontsize=16, fontweight='bold')
        
        # Plot 1: Box Loss
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Box Loss')
        axes[0, 0].set_title('Bounding Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Classification Loss
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Classification Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: mAP@0.5
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP@0.5')
        axes[1, 0].set_title('Mean Average Precision @ IoU=0.5')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Precision & Recall
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.figures_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Training curves saved to {save_path}")
    
    def plot_confusion_matrix(self):
        """Generate and plot confusion matrix."""
        logger.info("Generating confusion matrix...")
        
        # Find confusion matrix from validation
        results_dirs = [
            Path(self.output_config['results_dir']) / 'stage2_unfrozen',
            Path(self.output_config['results_dir']) / 'single_stage'
        ]
        
        confusion_matrix_path = None
        for results_dir in results_dirs:
            cm_path = results_dir / 'confusion_matrix.png'
            if cm_path.exists():
                confusion_matrix_path = cm_path
                break
        
        if confusion_matrix_path:
            # Copy to figures directory
            import shutil
            dest_path = self.figures_dir / 'confusion_matrix.png'
            shutil.copy2(confusion_matrix_path, dest_path)
            logger.info(f"✓ Confusion matrix copied to {dest_path}")
        else:
            logger.warning("Confusion matrix not found in results directory")
    
    def generate_sample_predictions(self, num_samples: int = 10):
        """
        Generate sample predictions with visualizations.
        
        Args:
            num_samples: Number of sample images to process
        """
        logger.info(f"Generating {num_samples} sample predictions...")
        
        # Get test images
        test_images_dir = Path(self.dataset_config['output_dir']) / 'test' / 'images'
        
        if not test_images_dir.exists():
            logger.warning(f"Test images directory not found: {test_images_dir}")
            return
        
        image_files = list(test_images_dir.glob('*.png'))[:num_samples]
        
        # Create predictions directory
        predictions_dir = Path(self.output_config['predictions_dir'])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(image_files, desc="Processing samples"):
            # Run inference
            results = self.model.predict(
                str(img_path),
                conf=self.model_config['conf_threshold'],
                iou=self.model_config['iou_threshold'],
                save=False
            )
            
            # Annotate image
            annotated = results[0].plot()
            
            # Save
            save_path = predictions_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(save_path), annotated)
        
        logger.info(f"✓ Sample predictions saved to {predictions_dir}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("\n" + "="*80)
        logger.info(" GENERATING EVALUATION REPORT")
        logger.info("="*80)
        
        # 1. Evaluate on test set
        metrics = self.evaluate_on_test_set()
        
        # 2. Plot training curves
        self.plot_training_curves()
        
        # 3. Copy confusion matrix
        self.plot_confusion_matrix()
        
        # 4. Generate sample predictions
        num_samples = self.eval_config.get('num_sample_predictions', 10)
        self.generate_sample_predictions(num_samples)
        
        logger.info("\n" + "="*80)
        logger.info(" EVALUATION REPORT COMPLETED")
        logger.info("="*80)
        logger.info(f"All outputs saved to: {self.figures_dir}")
        logger.info(f"Predictions saved to: {self.output_config['predictions_dir']}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(" PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Overall mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"Overall mAP@0.5:0.95: {metrics.box.map:.4f}")
        logger.info(f"Overall Precision: {metrics.box.mp:.4f}")
        logger.info(f"Overall Recall: {metrics.box.mr:.4f}")
        
        # Check for GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU Used: {gpu_name}")
        
        logger.info("="*80 + "\n")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained YOLOv8 model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(config_path=args.config)
    evaluator.generate_report()


if __name__ == "__main__":
    main()

"""
YOLOv8 Training Script with Transfer Learning
==============================================
Research-grade training pipeline with:
- Staged backbone freezing/unfreezing
- Mixed precision training
- Early stopping and validation-driven checkpointing
- Comprehensive logging

Author: Research Team
Date: February 2026
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOv8Trainer:
    """Manages YOLOv8 training with research-grade methodology."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.dataset_config = self.config['dataset']
        self.output_config = self.config['output']
        
        # Setup device
        self.device = self._setup_device()
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"Initialized trainer on device: {self.device}")
        
    def _setup_device(self) -> str:
        """
        Setup computation device.
        
        Returns:
            Device string ('cuda', '0', or 'cpu')
        """
        device_config = self.training_config['device']
        
        if device_config in ['cuda', '0', 0] and torch.cuda.is_available():
            device = '0'  # Use GPU 0
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            device = 'cpu'
            if device_config in ['cuda', '0', 0]:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            logger.info("Using CPU for training")
        
        return device
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.output_config['models_dir'],
            self.output_config['weights_dir'],
            self.output_config['results_dir'],
            self.output_config['logs_dir'],
            self.output_config['checkpoints_dir'],
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _freeze_layers(self, model, freeze: bool = True):
        """
        Freeze or unfreeze backbone layers.
        
        Args:
            model: YOLO model object
            freeze: If True, freeze backbone; if False, unfreeze all
        """
        try:
            if freeze:
                # Freeze backbone (first N layers)
                for name, param in model.model.named_parameters():
                    if 'model.0' in name or 'model.1' in name or 'model.2' in name:
                        param.requires_grad = False
                logger.info("✓ Backbone layers frozen")
            else:
                # Unfreeze all layers
                for param in model.model.parameters():
                    param.requires_grad = True
                logger.info("✓ All layers unfrozen")
        except AttributeError as e:
            logger.error(f"Error accessing model parameters: {e}")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Model attributes: {dir(model)}")
            raise
    
    def train_stage1_frozen(self, model: YOLO, data_yaml: str) -> YOLO:
        """
        Stage 1: Train with frozen backbone.
        
        Args:
            model: Pretrained YOLO model
            data_yaml: Path to data.yaml
            
        Returns:
            Trained model after stage 1
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Training with Frozen Backbone")
        logger.info("="*60)
        
        # Freeze backbone
        self._freeze_layers(model, freeze=True)
        
        # Training parameters for stage 1
        freeze_epochs = self.model_config['freeze_epochs']
        
        results = model.train(
            data=data_yaml,
            epochs=freeze_epochs,
            imgsz=self.model_config['img_size'],
            batch=self.model_config['batch_size'],
            device=self.device,
            
            # Optimizer
            optimizer=self.model_config['optimizer'],
            lr0=self.model_config['lr0'],
            lrf=self.model_config['lrf'],
            momentum=self.model_config['momentum'],
            weight_decay=self.model_config['weight_decay'],
            
            # Loss
            box=self.model_config['box_loss_gain'],
            cls=self.model_config['cls_loss_gain'],
            dfl=self.model_config['dfl_loss_gain'],
            
            # Regularization
            dropout=self.model_config['dropout'],
            label_smoothing=self.model_config['label_smoothing'],
            
            # Training settings
            workers=self.training_config['workers'],
            patience=0,  # No early stopping in stage 1
            save=True,
            save_period=self.training_config['save_period'],
            
            # Mixed precision
            amp=self.training_config['amp'],
            
            # Validation
            val=True,
            
            # Project organization
            project=self.output_config['results_dir'],
            name='stage1_frozen',
            exist_ok=True,
            
            # Visualization
            plots=True,
            verbose=True
        )
        
        logger.info(f"✓ Stage 1 completed: {freeze_epochs} epochs")
        
        # CRITICAL: Reload model from best stage 1 weights
        # After training, the model object's internal state is modified.
        # We MUST reload from weights to get a clean model for Stage 2.
        # Note: YOLO adds 'runs/detect/' prefix to the project path automatically
        best_stage1 = Path('runs') / 'detect' / self.output_config['results_dir'] / 'stage1_frozen' / 'weights' / 'best.pt'
        logger.info(f"Checking for best weights at: {best_stage1}")
        logger.info(f"Path exists: {best_stage1.exists()}")
        
        if not best_stage1.exists():
            logger.error(f"Best weights not found at {best_stage1}")
            logger.error("Training may have failed to save weights properly.")
            raise FileNotFoundError(f"Cannot find best.pt at {best_stage1}")
        
        logger.info(f"Loading best Stage 1 weights from: {best_stage1}")
        model = YOLO(str(best_stage1))
        logger.info("✓ Model reloaded successfully")
        
        return model
    
    def train_stage2_unfrozen(self, model: YOLO, data_yaml: str) -> YOLO:
        """
        Stage 2: Fine-tune with unfrozen backbone.
        
        Args:
            model: Model after stage 1
            data_yaml: Path to data.yaml
            
        Returns:
            Final trained model
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Fine-tuning with Unfrozen Backbone")
        logger.info("="*60)
        
        # Unfreeze all layers
        self._freeze_layers(model, freeze=False)
        
        # Training parameters for stage 2
        total_epochs = self.model_config['epochs']
        freeze_epochs = self.model_config['freeze_epochs']
        remaining_epochs = total_epochs - freeze_epochs
        
        results = model.train(
            data=data_yaml,
            epochs=remaining_epochs,
            imgsz=self.model_config['img_size'],
            batch=self.model_config['batch_size'],
            device=self.device,
            
            # Optimizer (slightly lower LR for fine-tuning)
            optimizer=self.model_config['optimizer'],
            lr0=self.model_config['lr0'] * 0.5,  # 50% LR for better fine-tuning
            lrf=self.model_config['lrf'],
            momentum=self.model_config['momentum'],
            weight_decay=self.model_config['weight_decay'],
            
            # Loss
            box=self.model_config['box_loss_gain'],
            cls=self.model_config['cls_loss_gain'],
            dfl=self.model_config['dfl_loss_gain'],
            
            # Regularization
            dropout=self.model_config['dropout'],
            label_smoothing=self.model_config['label_smoothing'],
            
            # Training settings
            workers=self.training_config['workers'],
            patience=self.model_config['patience'],  # Early stopping enabled
            save=True,
            save_period=self.training_config['save_period'],
            
            # Mixed precision
            amp=self.training_config['amp'],
            
            # Validation
            val=True,
            
            # Resume from stage 1
            resume=False,  # Start fresh counting
            
            # Project organization
            project=self.output_config['results_dir'],
            name='stage2_unfrozen',
            exist_ok=True,
            
            # Visualization
            plots=True,
            verbose=True
        )
        
        logger.info(f"✓ Stage 2 completed: {remaining_epochs} epochs")
        
        return model
    
    def train_single_stage(self, model: YOLO, data_yaml: str) -> YOLO:
        """
        Alternative: Single-stage training without freezing.
        
        Args:
            model: Pretrained YOLO model
            data_yaml: Path to data.yaml
            
        Returns:
            Trained model
        """
        logger.info("\n" + "="*60)
        logger.info("SINGLE-STAGE: Training without Freezing")
        logger.info("="*60)
        
        results = model.train(
            data=data_yaml,
            epochs=self.model_config['epochs'],
            imgsz=self.model_config['img_size'],
            batch=self.model_config['batch_size'],
            device=self.device,
            
            # Optimizer
            optimizer=self.model_config['optimizer'],
            lr0=self.model_config['lr0'],
            lrf=self.model_config['lrf'],
            momentum=self.model_config['momentum'],
            weight_decay=self.model_config['weight_decay'],
            
            # Loss
            box=self.model_config['box_loss_gain'],
            cls=self.model_config['cls_loss_gain'],
            dfl=self.model_config['dfl_loss_gain'],
            
            # Regularization
            dropout=self.model_config['dropout'],
            label_smoothing=self.model_config['label_smoothing'],
            
            # Training settings
            workers=self.training_config['workers'],
            patience=self.model_config['patience'],
            save=True,
            save_period=self.training_config['save_period'],
            
            # Mixed precision
            amp=self.training_config['amp'],
            
            # Validation
            val=True,
            
            # Project organization
            project=self.output_config['results_dir'],
            name='single_stage',
            exist_ok=True,
            
            # Visualization
            plots=True,
            verbose=True
        )
        
        logger.info(f"✓ Single-stage training completed: {self.model_config['epochs']} epochs")
        
        return model
    
    def run(self, use_staged_training: bool = True):
        """
        Execute complete training pipeline.
        
        Args:
            use_staged_training: If True, use 2-stage training; else single-stage
        """
        logger.info("\n" + "="*80)
        logger.info(" SMART PARKING YOLOV8 TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Configuration: {self.config['project']['name']} v{self.config['project']['version']}")
        
        # Check for data.yaml
        data_yaml = Path(self.dataset_config['output_dir']) / 'data.yaml'
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at {data_yaml}")
            logger.error("Please run data preprocessing first: python src/data_preparation/convert_annotations.py")
            return
        
        logger.info(f"Dataset: {data_yaml}")
        
        # Load pretrained model
        architecture = self.model_config['architecture']
        logger.info(f"Loading pretrained {architecture} model...")
        model = YOLO(f"{architecture}.pt")
        
        # Training
        if use_staged_training and self.model_config['freeze_backbone']:
            # Two-stage training
            model = self.train_stage1_frozen(model, str(data_yaml))
            model = self.train_stage2_unfrozen(model, str(data_yaml))
            final_weights = Path('runs') / 'detect' / self.output_config['results_dir'] / 'stage2_unfrozen' / 'weights' / 'best.pt'
        else:
            # Single-stage training
            model = self.train_single_stage(model, str(data_yaml))
            final_weights = Path('runs') / 'detect' / self.output_config['results_dir'] / 'single_stage' / 'weights' / 'best.pt'
        
        # Copy best weights to models directory
        best_model_path = Path(self.output_config['models_dir']) / 'best.pt'
        if final_weights.exists():
            import shutil
            shutil.copy2(final_weights, best_model_path)
            logger.info(f"✓ Best model saved to: {best_model_path}")
        
        logger.info("\n" + "="*80)
        logger.info(" TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"Results directory: {self.output_config['results_dir']}")
        
        # Print next steps
        logger.info("\n" + "="*80)
        logger.info(" NEXT STEPS")
        logger.info("="*80)
        logger.info("1. Evaluate model: python src/evaluation/evaluate_model.py")
        logger.info("2. View TensorBoard logs: tensorboard --logdir runs")
        logger.info("3. Start inference API: python src/api/app.py")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for parking space detection')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--single-stage',
        action='store_true',
        help='Use single-stage training instead of 2-stage'
    )
    
    args = parser.parse_args()
    
    trainer = YOLOv8Trainer(config_path=args.config)
    trainer.run(use_staged_training=not args.single_stage)


if __name__ == "__main__":
    main()

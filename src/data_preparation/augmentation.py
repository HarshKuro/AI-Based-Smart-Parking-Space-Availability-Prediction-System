"""
Data Augmentation Pipeline
===========================
Research-grade augmentation for constrained parking lot datasets.
Designed to improve robustness under illumination changes, occlusions,
and perspective distortions.

Author: Research Team
Date: February 2026
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import logging

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """Creates augmentation pipeline based on configuration."""
    
    def __init__(self, config_path: str = "config.yaml", training: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            config_path: Path to configuration file
            training: If True, applies augmentations; if False, only resizes
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['augmentation']
        self.dataset_config = self.config['dataset']
        self.training = training
        
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self) -> A.Compose:
        """
        Create Albumentations pipeline.
        
        Returns:
            Albumentations Compose object
        """
        target_size = self.dataset_config['target_size']
        
        if not self.training or not self.aug_config['enable']:
            # Validation/Test pipeline - no augmentation
            transforms = [
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=0,
                    value=(114, 114, 114)
                ),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ]
        else:
            # Training pipeline with augmentations
            prob = self.aug_config['probability']
            
            transforms = [
                # Resize
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=0,
                    value=(114, 114, 114)
                ),
                
                # Illumination & Exposure
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=self.aug_config['brightness_limit'],
                        contrast_limit=self.aug_config['contrast_limit'],
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=15,
                        p=1.0
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                ], p=prob),
                
                # Geometric Transformations
                A.ShiftScaleRotate(
                    shift_limit=self.aug_config['shift_limit'],
                    scale_limit=self.aug_config['scale_limit'],
                    rotate_limit=self.aug_config['rotate_limit'],
                    border_mode=0,
                    value=(114, 114, 114),
                    p=prob
                ),
                
                # Horizontal flip (parking lots often symmetric)
                A.HorizontalFlip(p=0.5),
                
                # Occlusion Simulation
                A.CoarseDropout(
                    max_holes=self.aug_config['coarse_dropout']['max_holes'],
                    max_height=self.aug_config['coarse_dropout']['max_height'],
                    max_width=self.aug_config['coarse_dropout']['max_width'],
                    min_holes=self.aug_config['coarse_dropout']['min_holes'],
                    min_height=self.aug_config['coarse_dropout']['min_height'],
                    min_width=self.aug_config['coarse_dropout']['min_width'],
                    fill_value=self.aug_config['coarse_dropout']['fill_value'],
                    p=prob * 0.5
                ),
                
                # Weather Conditions
                A.OneOf([
                    A.GaussianBlur(
                        blur_limit=self.aug_config['blur_limit'],
                        p=1.0
                    ),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=prob * 0.3),
                
                # Shadow simulation (important for outdoor parking)
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    shadow_dimension=5,
                    p=prob * 0.4
                ),
                
                # Normalize
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ]
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3  # Keep boxes with >30% visibility
            )
        )
    
    def __call__(self, image, bboxes=None, class_labels=None):
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class IDs
            
        Returns:
            Augmented image and (optionally) transformed bboxes
        """
        if bboxes is None or class_labels is None:
            # No bounding boxes (inference mode)
            transformed = self.pipeline(image=image)
            return transformed['image']
        else:
            # With bounding boxes (training mode)
            transformed = self.pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']


def get_train_transforms(config_path: str = "config.yaml"):
    """Get training augmentation pipeline."""
    return AugmentationPipeline(config_path=config_path, training=True)


def get_val_transforms(config_path: str = "config.yaml"):
    """Get validation augmentation pipeline (no augmentation)."""
    return AugmentationPipeline(config_path=config_path, training=False)


if __name__ == "__main__":
    import cv2
    import numpy as np
    
    # Test augmentation pipeline
    logger.info("Testing augmentation pipeline...")
    
    # Create dummy image
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    bboxes = [[0.5, 0.5, 0.2, 0.3]]  # YOLO format
    class_labels = [0]
    
    # Create pipelines
    train_aug = get_train_transforms()
    val_aug = get_val_transforms()
    
    # Apply training augmentation
    aug_img, aug_boxes, aug_labels = train_aug(image, bboxes, class_labels)
    logger.info(f"Training augmentation: Image shape={aug_img.shape}, Boxes={len(aug_boxes)}")
    
    # Apply validation augmentation
    val_img = val_aug(image)
    logger.info(f"Validation augmentation: Image shape={val_img.shape}")
    
    logger.info("✓ Augmentation pipeline test completed")

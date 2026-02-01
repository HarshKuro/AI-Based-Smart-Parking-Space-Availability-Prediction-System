"""
Data Preprocessing Module - XML to YOLO Conversion
==================================================
Converts CVAT XML polygon annotations to YOLO bounding box format.
Handles multi-class parking space detection with stratified splitting.

Author: Research Team
Date: February 2026
"""

import os
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationConverter:
    """Converts CVAT XML polygon annotations to YOLO format."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize converter with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.classes = self.dataset_config['classes']
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Paths
        self.root_dir = Path(self.dataset_config['root_dir'])
        self.images_dir = Path(self.dataset_config['images_dir'])
        self.annotations_xml = Path(self.dataset_config['annotations_xml'])
        self.output_dir = Path(self.dataset_config['output_dir'])
        
        logger.info(f"Initialized converter with {len(self.classes)} classes")
        
    def polygon_to_bbox(self, points: str) -> Tuple[float, float, float, float]:
        """
        Convert polygon points to bounding box.
        
        Args:
            points: Semicolon-separated polygon coordinates "x1,y1;x2,y2;..."
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        coords = []
        for point in points.split(';'):
            x, y = map(float, point.split(','))
            coords.append([x, y])
        
        coords = np.array(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        return x_min, y_min, x_max, y_max
    
    def normalize_bbox(self, bbox: Tuple[float, float, float, float], 
                       img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert absolute bbox to YOLO normalized format.
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            YOLO format: (x_center, y_center, width, height) - all normalized [0, 1]
        """
        x_min, y_min, x_max, y_max = bbox
        
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Clip to [0, 1] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return x_center, y_center, width, height
    
    def parse_xml_annotations(self) -> Dict[str, List[Dict]]:
        """
        Parse CVAT XML file and extract annotations.
        
        Returns:
            Dictionary mapping image names to list of annotations
            {
                "0.png": [
                    {"class": "free_parking_space", "bbox": (x, y, w, h)},
                    ...
                ]
            }
        """
        tree = ET.parse(self.annotations_xml)
        root = tree.getroot()
        
        annotations = {}
        
        for image in tqdm(root.findall('.//image'), desc="Parsing XML"):
            img_name = Path(image.get('name')).name
            img_width = int(image.get('width'))
            img_height = int(image.get('height'))
            
            annotations[img_name] = []
            
            for polygon in image.findall('.//polygon'):
                label = polygon.get('label')
                
                if label not in self.class_to_id:
                    logger.warning(f"Unknown class '{label}' in image {img_name}")
                    continue
                
                points = polygon.get('points')
                
                # Convert polygon to bbox
                bbox_abs = self.polygon_to_bbox(points)
                bbox_norm = self.normalize_bbox(bbox_abs, img_width, img_height)
                
                annotations[img_name].append({
                    'class': label,
                    'class_id': self.class_to_id[label],
                    'bbox': bbox_norm
                })
        
        logger.info(f"Parsed annotations for {len(annotations)} images")
        return annotations
    
    def create_yolo_dataset(self, annotations: Dict[str, List[Dict]]):
        """
        Create YOLO format dataset structure.
        
        Args:
            annotations: Parsed annotations from XML
        """
        # Create output directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all image names
        image_names = list(annotations.keys())
        
        # Stratified split by class distribution
        # First, create a label for each image based on majority class
        image_labels = []
        for img_name in image_names:
            if not annotations[img_name]:
                image_labels.append(-1)  # No annotations
                continue
            
            class_counts = {}
            for ann in annotations[img_name]:
                cls_id = ann['class_id']
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            
            majority_class = max(class_counts, key=class_counts.get)
            image_labels.append(majority_class)
        
        # Split: train/temp (70/30), then temp into val/test (66/34 ≈ 20/10 of total)
        train_ratio = self.dataset_config['train_ratio']
        val_ratio = self.dataset_config['val_ratio']
        test_ratio = self.dataset_config['test_ratio']
        random_seed = self.dataset_config['random_seed']
        
        # Adjust for two-step split
        temp_ratio = val_ratio + test_ratio
        val_of_temp_ratio = val_ratio / temp_ratio
        
        # Check if stratification is possible (need at least 2 samples per class)
        from collections import Counter
        label_counts = Counter(image_labels)
        min_samples = min(label_counts.values())
        
        try:
            if min_samples < 2:
                logger.warning(f"Insufficient samples for stratification (min={min_samples}). Using random split.")
                stratify_train = None
                stratify_val = None
            else:
                stratify_train = image_labels
                stratify_val = None  # Will be set later
            
            train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
                image_names, image_labels,
                test_size=temp_ratio,
                random_state=random_seed,
                stratify=stratify_train
            )
            
            # Check if stratification is possible for val/test split
            temp_label_counts = Counter(temp_lbls)
            temp_min_samples = min(temp_label_counts.values())
            
            if temp_min_samples >= 2:
                stratify_val = temp_lbls
            else:
                logger.warning(f"Insufficient samples for val/test stratification (min={temp_min_samples}). Using random split.")
                stratify_val = None
            
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                test_size=(1 - val_of_temp_ratio),
                random_state=random_seed,
                stratify=stratify_val
            )
        except ValueError as e:
            logger.warning(f"Stratification failed: {e}. Falling back to random split.")
            train_imgs, temp_imgs = train_test_split(
                image_names,
                test_size=temp_ratio,
                random_state=random_seed
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                test_size=(1 - val_of_temp_ratio),
                random_state=random_seed
            )
        
        split_images = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        logger.info(f"Dataset split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        # Copy images and create label files
        for split, img_list in split_images.items():
            logger.info(f"Processing {split} split...")
            
            for img_name in tqdm(img_list, desc=f"Creating {split} set"):
                # Copy image
                src_img = self.images_dir / img_name
                dst_img = self.output_dir / split / 'images' / img_name
                
                if not src_img.exists():
                    logger.warning(f"Image not found: {src_img}")
                    continue
                
                shutil.copy2(src_img, dst_img)
                
                # Create label file
                label_file = self.output_dir / split / 'labels' / img_name.replace('.png', '.txt')
                
                with open(label_file, 'w') as f:
                    for ann in annotations[img_name]:
                        class_id = ann['class_id']
                        x, y, w, h = ann['bbox']
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        # Create data.yaml for YOLOv8
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {self.output_dir / 'data.yaml'}")
        
    def generate_statistics(self, annotations: Dict[str, List[Dict]]):
        """
        Generate dataset statistics.
        
        Args:
            annotations: Parsed annotations
        """
        total_images = len(annotations)
        total_annotations = sum(len(anns) for anns in annotations.values())
        
        class_counts = {cls: 0 for cls in self.classes}
        for anns in annotations.values():
            for ann in anns:
                class_counts[ann['class']] += 1
        
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Images: {total_images}")
        logger.info(f"Total Annotations: {total_annotations}")
        logger.info(f"Average annotations per image: {total_annotations/total_images:.2f}")
        logger.info("\nClass Distribution:")
        for cls, count in class_counts.items():
            percentage = (count / total_annotations) * 100
            logger.info(f"  {cls}: {count} ({percentage:.2f}%)")
        logger.info("="*60 + "\n")
    
    def run(self):
        """Execute full conversion pipeline."""
        logger.info("Starting annotation conversion pipeline...")
        
        # Parse XML
        annotations = self.parse_xml_annotations()
        
        # Generate statistics
        self.generate_statistics(annotations)
        
        # Create YOLO dataset
        self.create_yolo_dataset(annotations)
        
        logger.info("✓ Conversion completed successfully!")
        logger.info(f"Output directory: {self.output_dir.absolute()}")


def main():
    """Main execution function."""
    converter = AnnotationConverter()
    converter.run()


if __name__ == "__main__":
    main()

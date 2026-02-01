"""
Merge Multiple Datasets
=======================
Combines dataset/ (CVAT XML) and data2/ (JSON bbox) into unified YOLO format.

Author: Research Team  
Date: February 2026
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm

class DatasetMerger:
    """Merge multiple annotation formats into YOLO format."""
    
    def __init__(self):
        self.dataset1_dir = Path("dataset")
        self.dataset2_dir = Path("data2")
        self.output_dir = Path("data_merged")
        
        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Class mapping (2 classes only: free and occupied)
        self.class_map = {
            "free_parking_space": 0,
            "Free": 0,
            "not_free_parking_space": 1,
            "Occupied": 1
        }
        
        # Exclude partially_free_parking_space (insufficient training samples)
        self.excluded_classes = ["partially_free_parking_space", "Partial"]
        
        self.class_names = ["free_parking_space", "not_free_parking_space"]
        
    def polygon_to_bbox(self, points_str, img_width, img_height):
        """Convert polygon points to YOLO bbox format."""
        points = [float(x) for x in points_str.replace(';', ',').split(',')]
        xs = points[0::2]
        ys = points[1::2]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return x_center, y_center, width, height
    
    def process_dataset1_cvat(self):
        """Process dataset 1 (CVAT XML format)."""
        print("\n📦 Processing Dataset 1 (CVAT XML)...")
        
        xml_file = self.dataset1_dir / "annotations.xml"
        if not xml_file.exists():
            print("  ⚠️  annotations.xml not found!")
            return 0
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        count = 0
        for image in tqdm(root.findall('image'), desc="  Converting"):
            img_name = image.get('name')
            img_id = Path(img_name).stem
            img_width = int(image.get('width'))
            img_height = int(image.get('height'))
            
            # Find actual image file (try .png and .jpg)
            src_img = None
            for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                potential_img = self.dataset1_dir / "images" / f"{img_id}{ext}"
                if potential_img.exists():
                    src_img = potential_img
                    break
            
            if src_img:
                # Determine output extension
                out_ext = src_img.suffix
                dst_img = self.output_dir / "images" / f"ds1_{img_id}{out_ext}"
                shutil.copy2(src_img, dst_img)
                
                # Process annotations
                labels = []
                for polygon in image.findall('polygon'):
                    label = polygon.get('label')
                    points = polygon.get('points')
                    
                    # Skip excluded classes
                    if label in self.excluded_classes:
                        continue
                    
                    if label in self.class_map:
                        class_id = self.class_map[label]
                        x_center, y_center, width, height = self.polygon_to_bbox(
                            points, img_width, img_height
                        )
                        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save labels
                if labels:
                    label_file = self.output_dir / "labels" / f"ds1_{img_id}.txt"
                    label_file.write_text('\n'.join(labels))
                    count += 1
        
        print(f"  ✓ Processed {count} images from dataset 1")
        return count
    
    def process_dataset2_json(self):
        """Process dataset 2 (JSON bbox format)."""
        print("\n📦 Processing Dataset 2 (JSON Bounding Boxes)...")
        
        if not self.dataset2_dir.exists():
            print("  ⚠️  data2/ directory not found!")
            return 0
        
        json_files = list(self.dataset2_dir.glob("*.json"))
        count = 0
        
        for json_file in tqdm(json_files, desc="  Converting"):
            img_id = json_file.stem
            img_file = self.dataset2_dir / f"{img_id}.jpg"
            
            if not img_file.exists():
                continue
            
            # Load image to get dimensions
            img = Image.open(img_file)
            img_width, img_height = img.size
            
            # Copy image with prefix
            dst_img = self.output_dir / "images" / f"ds2_{img_id}.jpg"
            shutil.copy2(img_file, dst_img)
            
            # Load annotations
            with open(json_file) as f:
                data = json.load(f)
            
            labels = []
            for bbox in data.get('labels', []):
                label = bbox.get('name')
                
                # Skip excluded classes
                if label in self.excluded_classes:
                    continue
                    
                if label not in self.class_map:
                    continue
                
                class_id = self.class_map[label]
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save labels
            if labels:
                label_file = self.output_dir / "labels" / f"ds2_{img_id}.txt"
                label_file.write_text('\n'.join(labels))
                count += 1
        
        print(f"  ✓ Processed {count} images from dataset 2")
        return count
    
    def create_data_yaml(self, train_count, val_count, test_count):
        """Create data.yaml for training."""
        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,
            'names': self.class_names
        }
        
        yaml_file = self.output_dir / 'data.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"\n✓ Created data.yaml at {yaml_file}")
    
    def split_dataset(self, train_ratio=0.70, val_ratio=0.20):
        """Split merged dataset into train/val/test."""
        print("\n📊 Splitting merged dataset...")
        
        from sklearn.model_selection import train_test_split
        
        # Get all image files
        all_images = list((self.output_dir / "images").glob("*"))
        all_images = [img.name for img in all_images]
        
        # Split
        train_imgs, temp_imgs = train_test_split(
            all_images, train_size=train_ratio, random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=val_ratio/(1-train_ratio), random_state=42
        )
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Move files to split directories
        def move_to_split(img_list, split_name):
            for img_name in tqdm(img_list, desc=f"  {split_name}"):
                img_stem = Path(img_name).stem
                
                # Move image
                src_img = self.output_dir / 'images' / img_name
                dst_img = self.output_dir / 'images' / split_name / img_name
                shutil.move(str(src_img), str(dst_img))
                
                # Move label
                src_label = self.output_dir / 'labels' / f'{img_stem}.txt'
                if src_label.exists():
                    dst_label = self.output_dir / 'labels' / split_name / f'{img_stem}.txt'
                    shutil.move(str(src_label), str(dst_label))
        
        move_to_split(train_imgs, 'train')
        move_to_split(val_imgs, 'val')
        move_to_split(test_imgs, 'test')
        
        print(f"\n  ✓ Train: {len(train_imgs)} images ({train_ratio*100:.0f}%)")
        print(f"  ✓ Val: {len(val_imgs)} images ({val_ratio*100:.0f}%)")
        print(f"  ✓ Test: {len(test_imgs)} images ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return len(train_imgs), len(val_imgs), len(test_imgs)
    
    def analyze_merged_dataset(self):
        """Analyze class distribution in merged dataset."""
        print("\n📈 Analyzing merged dataset...")
        
        class_counts = {0: 0, 1: 0}
        total_annotations = 0
        
        for label_file in (self.output_dir / "labels").glob("**/*.txt"):
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            total_annotations += 1
        
        print("\n  Class Distribution:")
        for class_id, count in class_counts.items():
            class_name = self.class_names[class_id]
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            status = "✓" if count > 50 else "⚠️"
            print(f"    {status} {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n  Total annotations: {total_annotations}")
        print(f"  ⚠️  Excluded partially_free_parking_space (only 6 samples - insufficient for training)")
    
    def merge_all(self):
        """Execute complete merging pipeline."""
        print("\n" + "="*60)
        print("  MERGING DATASETS")
        print("="*60)
        
        count1 = self.process_dataset1_cvat()
        count2 = self.process_dataset2_json()
        
        total = count1 + count2
        print(f"\n✓ Total images merged: {total}")
        
        # Split dataset
        train_count, val_count, test_count = self.split_dataset()
        
        # Create data.yaml
        self.create_data_yaml(train_count, val_count, test_count)
        
        # Analyze
        self.analyze_merged_dataset()
        
        print("\n" + "="*60)
        print("  ✓ DATASET MERGING COMPLETE")
        print("="*60)
        print(f"\n  Output directory: {self.output_dir.absolute()}")
        print(f"  Next step: Update config.yaml to use 'data_merged'")


if __name__ == "__main__":
    merger = DatasetMerger()
    merger.merge_all()

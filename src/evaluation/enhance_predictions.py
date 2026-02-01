"""
Create Beautiful Prediction Visualizations
==========================================
Enhance prediction images with better formatting and annotations.

Author: Research Team
Date: February 2026
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class PredictionVisualizer:
    """Create publication-quality prediction visualizations."""
    
    def __init__(self):
        self.model_path = Path('models/best.pt')
        self.test_images = Path('data_processed/test/images')
        self.output_dir = Path('figures/predictions_enhanced')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Class names and colors
        self.class_names = ['Free', 'Occupied', 'Partially Free']
        self.colors = {
            0: (46, 204, 113),   # Green for free
            1: (231, 76, 60),    # Red for occupied
            2: (241, 196, 15)    # Yellow for partially free
        }
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        
    def create_enhanced_predictions(self):
        """Create enhanced prediction visualizations."""
        print("\nGenerating enhanced predictions...")
        
        # Get all test images
        image_files = sorted(list(self.test_images.glob('*.png')) + 
                           list(self.test_images.glob('*.jpg')))
        
        for img_path in image_files:
            print(f"Processing {img_path.name}...")
            
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(img, conf=0.25, verbose=False)[0]
            
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Original image
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Predictions
            axes[1].imshow(img_rgb)
            
            # Count detections per class
            class_counts = {0: 0, 1: 0, 2: 0}
            
            # Draw bounding boxes
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                class_counts[cls] += 1
                
                # Convert color from BGR to RGB
                color_bgr = self.colors[cls]
                color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2.5, edgecolor=color_rgb, facecolor='none'
                )
                axes[1].add_patch(rect)
                
                # Add label with confidence
                label = f'{self.class_names[cls]} {conf:.2f}'
                axes[1].text(x1, y1-5, label,
                           bbox=dict(facecolor=color_rgb, alpha=0.8, edgecolor='none'),
                           fontsize=9, color='white', fontweight='bold')
            
            # Title with statistics
            total = sum(class_counts.values())
            title = (f'Detections: {total} total | '
                    f'{class_counts[0]} Free | '
                    f'{class_counts[1]} Occupied | '
                    f'{class_counts[2]} Partially Free')
            axes[1].set_title(title, fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Save
            output_path = self.output_dir / f'{img_path.stem}_enhanced.png'
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {output_path.name}")
            
    def create_grid_visualization(self):
        """Create a grid of all predictions."""
        print("\nGenerating grid visualization...")
        
        # Get all enhanced images
        enhanced_images = sorted(list(self.output_dir.glob('*_enhanced.png')))
        
        if not enhanced_images:
            print("No enhanced images found!")
            return
            
        n_images = len(enhanced_images)
        cols = 2
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10*rows))
        fig.suptitle('Smart Parking Detection Results - All Test Images', 
                    fontsize=18, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_path in enumerate(enhanced_images):
            row = idx // cols
            col = idx % cols
            
            img = plt.imread(str(img_path))
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Image {idx+1}', fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir.parent / 'all_predictions_grid.png', 
                   dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir.parent / 'all_predictions_grid.png'}")
        plt.close()
        
    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("GENERATING ENHANCED PREDICTION VISUALIZATIONS")
        print("="*60)
        
        self.create_enhanced_predictions()
        self.create_grid_visualization()
        
        print("\n" + "="*60)
        print("✓ ALL PREDICTION VISUALIZATIONS GENERATED")
        print("="*60)
        print(f"Output directory: {self.output_dir.absolute()}")


if __name__ == "__main__":
    visualizer = PredictionVisualizer()
    visualizer.generate_all()

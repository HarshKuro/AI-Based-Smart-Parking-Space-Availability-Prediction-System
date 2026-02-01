"""
Create Beautiful Visualizations for Training Results
====================================================
Generates publication-quality training curves, confusion matrices,
and performance comparison charts.

Author: Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class VisualizationGenerator:
    """Generate comprehensive training visualizations."""
    
    def __init__(self):
        self.stage1_dir = Path('runs/detect/results/stage1_frozen')
        self.stage2_dir = Path('runs/detect/results/stage2_unfrozen')
        self.output_dir = Path('figures')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_results(self):
        """Load training results from CSV files."""
        stage1_csv = self.stage1_dir / 'results.csv'
        stage2_csv = self.stage2_dir / 'results.csv'
        
        dfs = []
        
        if stage1_csv.exists():
            df1 = pd.read_csv(stage1_csv)
            df1.columns = df1.columns.str.strip()
            df1['stage'] = 'Stage 1 (Frozen)'
            df1['global_epoch'] = df1['epoch']
            dfs.append(df1)
            
        if stage2_csv.exists():
            df2 = pd.read_csv(stage2_csv)
            df2.columns = df2.columns.str.strip()
            df2['stage'] = 'Stage 2 (Unfrozen)'
            # Offset epoch numbers for stage 2
            stage1_epochs = len(df1) if dfs else 0
            df2['global_epoch'] = df2['epoch'] + stage1_epochs
            dfs.append(df2)
            
        if not dfs:
            print("No results.csv found!")
            return None
            
        return pd.concat(dfs, ignore_index=True)
    
    def plot_training_curves(self, df):
        """Create comprehensive training curve plots."""
        print("Generating training curves...")
        
        # 1. Loss Curves (Box, Class, DFL)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress: Loss Metrics', fontsize=16, fontweight='bold')
        
        # Box Loss
        ax = axes[0, 0]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            ax.plot(stage_data['global_epoch'], stage_data['train/box_loss'], 
                   label=f'{stage} (Train)', linewidth=2)
            if 'val/box_loss' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['val/box_loss'], 
                       label=f'{stage} (Val)', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Box Loss')
        ax.set_title('Bounding Box Regression Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Class Loss
        ax = axes[0, 1]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            ax.plot(stage_data['global_epoch'], stage_data['train/cls_loss'], 
                   label=f'{stage} (Train)', linewidth=2)
            if 'val/cls_loss' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['val/cls_loss'], 
                       label=f'{stage} (Val)', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Classification Loss')
        ax.set_title('Classification Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # DFL Loss
        ax = axes[1, 0]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            ax.plot(stage_data['global_epoch'], stage_data['train/dfl_loss'], 
                   label=f'{stage} (Train)', linewidth=2)
            if 'val/dfl_loss' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['val/dfl_loss'], 
                       label=f'{stage} (Val)', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('DFL Loss')
        ax.set_title('Distribution Focal Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Combined Loss
        ax = axes[1, 1]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            # Calculate total loss
            total_loss = (stage_data['train/box_loss'] + 
                         stage_data['train/cls_loss'] + 
                         stage_data['train/dfl_loss'])
            ax.plot(stage_data['global_epoch'], total_loss, 
                   label=f'{stage}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Combined Training Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_loss_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'training_loss_curves.png'}")
        plt.close()
        
        # 2. Performance Metrics (Precision, Recall, mAP)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress: Performance Metrics', fontsize=16, fontweight='bold')
        
        # Precision
        ax = axes[0, 0]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            if 'metrics/precision(B)' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['metrics/precision(B)'], 
                       label=stage, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.set_title('Precision @ IoU=0.5', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Recall
        ax = axes[0, 1]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            if 'metrics/recall(B)' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['metrics/recall(B)'], 
                       label=stage, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.set_title('Recall @ IoU=0.5', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # mAP@0.5
        ax = axes[1, 0]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            if 'metrics/mAP50(B)' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['metrics/mAP50(B)'], 
                       label=stage, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5')
        ax.set_title('Mean Average Precision @ IoU=0.5', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # mAP@0.5:0.95
        ax = axes[1, 1]
        for stage in df['stage'].unique():
            stage_data = df[df['stage'] == stage]
            if 'metrics/mAP50-95(B)' in stage_data.columns:
                ax.plot(stage_data['global_epoch'], stage_data['metrics/mAP50-95(B)'], 
                       label=stage, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5:0.95')
        ax.set_title('Mean Average Precision @ IoU=0.5:0.95', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_metrics_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'training_metrics_curves.png'}")
        plt.close()
        
    def plot_performance_comparison(self):
        """Create performance comparison bar charts."""
        print("Generating performance comparison...")
        
        # Load test metrics
        metrics_file = self.output_dir / 'test_metrics.csv'
        if not metrics_file.exists():
            print("Test metrics not found!")
            return
            
        df = pd.read_csv(metrics_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance on Test Set', fontsize=16, fontweight='bold')
        
        # Per-class metrics
        ax = axes[0]
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['mAP@0.5'], width, label='mAP@0.5', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['class'].str.replace('_', ' ').str.title(), rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (p, r, m) in enumerate(zip(df['precision'], df['recall'], df['mAP@0.5'])):
            ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, m + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Overall metrics
        ax = axes[1]
        overall = df[df['class'] == 'Overall'].iloc[0]
        metrics = {
            'Precision': overall['precision'],
            'Recall': overall['recall'],
            'mAP@0.5': overall['mAP@0.5'],
            'mAP@0.5:0.95': overall['mAP@0.5:0.95']
        }
        
        bars = ax.bar(metrics.keys(), metrics.values(), alpha=0.8, 
                     color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Score')
        ax.set_title('Overall Model Performance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'performance_comparison.png'}")
        plt.close()
        
    def create_results_dashboard(self):
        """Create a comprehensive results dashboard."""
        print("Generating results dashboard...")
        
        metrics_file = self.output_dir / 'test_metrics.csv'
        if not metrics_file.exists():
            return
            
        df = pd.read_csv(metrics_file)
        overall = df[df['class'] == 'Overall'].iloc[0]
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Smart Parking System - Training Results Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Overall Metrics (Large center)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        metrics = {
            'mAP@0.5': overall['mAP@0.5'],
            'mAP@0.5:0.95': overall['mAP@0.5:0.95'],
            'Precision': overall['precision'],
            'Recall': overall['recall']
        }
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        bars = ax1.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.8)
        ax1.set_xlabel('Score', fontweight='bold')
        ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 1])
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f} ({width*100:.1f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=12)
        
        # 2. Per-class breakdown
        ax2 = fig.add_subplot(gs[0, 2])
        class_data = df[df['class'] != 'Overall']
        ax2.axis('tight')
        ax2.axis('off')
        table_data = []
        for _, row in class_data.iterrows():
            table_data.append([
                row['class'].replace('_', ' ').title()[:20],
                f"{row['mAP@0.5']:.2f}",
                f"{row['precision']:.2f}",
                f"{row['recall']:.2f}"
            ])
        table = ax2.table(cellText=table_data,
                         colLabels=['Class', 'mAP', 'Prec', 'Rec'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.5, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax2.set_title('Per-Class Results', fontsize=12, fontweight='bold', pad=20)
        
        # 3. Training summary
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        summary_text = [
            "Training Configuration:",
            "• Architecture: YOLOv8n",
            "• Total Epochs: 140 (10+130)",
            "• Stage 1: Frozen backbone",
            "• Stage 2: Full fine-tuning",
            "• Batch Size: 8",
            "• Image Size: 640x640",
            "• Optimizer: AdamW",
            "• Best Epoch: 48 (Stage 2)",
            "",
            "Dataset Split:",
            "• Train: 20 images",
            "• Val: 6 images",
            "• Test: 4 images",
        ]
        ax3.text(0.1, 0.9, '\n'.join(summary_text), transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax3.set_title('Training Summary', fontsize=12, fontweight='bold')
        
        # 4. Key achievements
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        achievements = [
            f"🎯 Achieved {overall['mAP@0.5']*100:.1f}% mAP@0.5 on test set",
            f"✓ Free parking spaces: {df[df['class']=='free_parking_space']['mAP@0.5'].values[0]*100:.1f}% mAP",
            f"✓ Occupied spaces: {df[df['class']=='not_free_parking_space']['mAP@0.5'].values[0]*100:.1f}% mAP",
            f"⚡ Inference speed: ~66ms per image on CPU",
            f"📊 Model size: 6.2 MB (3M parameters)",
        ]
        ax4.text(0.5, 0.5, '  |  '.join(achievements), transform=ax4.transAxes,
                fontsize=12, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
        
        plt.savefig(self.output_dir / 'results_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'results_dashboard.png'}")
        plt.close()
        
    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*60)
        
        # Load training data
        df = self.load_results()
        
        if df is not None:
            self.plot_training_curves(df)
        
        self.plot_performance_comparison()
        self.create_results_dashboard()
        
        print("\n" + "="*60)
        print("✓ ALL VISUALIZATIONS GENERATED")
        print("="*60)
        print(f"Output directory: {self.output_dir.absolute()}")


if __name__ == "__main__":
    generator = VisualizationGenerator()
    generator.generate_all()

"""
Generate comprehensive learning analysis graphs for 2-class model
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
output_dir = Path('research-2-class/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print(" GENERATING LEARNING ANALYSIS GRAPHS FOR 2-CLASS MODEL")
print("="*80 + "\n")

# Load training data
stage1_results = pd.read_csv('runs/detect/results/stage1_frozen/results.csv')
stage2_results = pd.read_csv('runs/detect/results/stage2_unfrozen/results.csv')

stage1_results = stage1_results.iloc[:, :12].copy()  # Include val/dfl_loss
stage1_results.columns = stage1_results.columns.str.strip()

stage2_results = stage2_results.iloc[:, :12].copy()  # Include val/dfl_loss
stage2_results.columns = stage2_results.columns.str.strip()

stage2_results['epoch'] = stage2_results['epoch'] + len(stage1_results)
df = pd.concat([stage1_results, stage2_results], ignore_index=True)

print("📊 1. Generating comprehensive learning curve analysis...")

# 1. Detailed Learning Curves
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('2-Class Model: Complete Learning Analysis\nYOLOv8n | 180 Images | Free vs Occupied Parking Detection', 
             fontsize=16, fontweight='bold', y=0.98)

# Loss curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', linewidth=2, alpha=0.8)
ax1.plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Box Loss', linewidth=2, alpha=0.8)
ax1.plot(df['epoch'], df['train/cls_loss'], 'g-', label='Train Class Loss', linewidth=2, alpha=0.8)
ax1.plot(df['val/cls_loss'], 'orange', linestyle='--', label='Val Class Loss', linewidth=2, alpha=0.8)
ax1.axvline(x=20, color='purple', linestyle=':', label='Stage Transition', linewidth=2.5)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training & Validation Loss Progression (Lower is Better)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# mAP progression
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', linewidth=3, label='mAP@0.5')
ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'b-', linewidth=2, label='mAP@0.5:0.95', alpha=0.7)
ax2.axvline(x=20, color='purple', linestyle=':', linewidth=2)
ax2.axhline(y=0.8360, color='red', linestyle='--', label='Final: 83.60%', linewidth=2)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('mAP Score', fontweight='bold')
ax2.set_title('Mean Average Precision Growth', fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Precision & Recall
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df['epoch'], df['metrics/precision(B)'], 'b-', linewidth=2.5, label='Precision')
ax3.plot(df['epoch'], df['metrics/recall(B)'], 'r-', linewidth=2.5, label='Recall')
ax3.axvline(x=20, color='purple', linestyle=':', linewidth=2)
ax3.fill_between(df['epoch'], df['metrics/precision(B)'], df['metrics/recall(B)'], 
                  alpha=0.2, color='green')
ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Precision & Recall Balance', fontsize=11, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1])

# Learning rate effect
ax4 = fig.add_subplot(gs[1, 2])
stage1_mAP = df['metrics/mAP50(B)'][:20]
stage2_mAP = df['metrics/mAP50(B)'][20:]
ax4.bar(range(len(stage1_mAP)), stage1_mAP, color='skyblue', label='Stage 1 (LR=0.01)', alpha=0.8)
ax4.bar(range(len(stage1_mAP), len(stage1_mAP) + len(stage2_mAP)), stage2_mAP, 
        color='coral', label='Stage 2 (LR=0.005)', alpha=0.8)
ax4.set_xlabel('Epoch', fontweight='bold')
ax4.set_ylabel('mAP@0.5', fontweight='bold')
ax4.set_title('Two-Stage Learning Impact', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Loss reduction rate
ax5 = fig.add_subplot(gs[2, 0])
total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
loss_reduction = (total_loss[0] - total_loss) / total_loss[0] * 100
ax5.plot(df['epoch'], loss_reduction, 'darkgreen', linewidth=3)
ax5.axvline(x=20, color='purple', linestyle=':', linewidth=2)
ax5.fill_between(df['epoch'], 0, loss_reduction, alpha=0.3, color='green')
ax5.set_xlabel('Epoch', fontweight='bold')
ax5.set_ylabel('Loss Reduction (%)', fontweight='bold')
ax5.set_title('Model Learning Progress (Cumulative)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Convergence analysis
ax6 = fig.add_subplot(gs[2, 1])
mAP_improvement = df['metrics/mAP50(B)'].diff().fillna(0) * 100
ax6.bar(df['epoch'], mAP_improvement, color=['red' if x < 0 else 'green' for x in mAP_improvement], 
        alpha=0.7, width=0.8)
ax6.axvline(x=20, color='purple', linestyle=':', linewidth=2)
ax6.axhline(y=0, color='black', linewidth=0.8)
ax6.set_xlabel('Epoch', fontweight='bold')
ax6.set_ylabel('mAP Change (%)', fontweight='bold')
ax6.set_title('Per-Epoch Learning Rate', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Training stability
ax7 = fig.add_subplot(gs[2, 2])
val_stability = (df['val/box_loss'].rolling(window=5).std() * 100).fillna(0)
ax7.plot(df['epoch'], val_stability, 'purple', linewidth=2.5)
ax7.axvline(x=20, color='purple', linestyle=':', linewidth=2)
ax7.fill_between(df['epoch'], 0, val_stability, alpha=0.3, color='purple')
ax7.set_xlabel('Epoch', fontweight='bold')
ax7.set_ylabel('Loss Std Dev (5-epoch window)', fontweight='bold')
ax7.set_title('Training Stability Analysis', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.savefig(output_dir / 'learning_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print(f"   ✓ learning_analysis_comprehensive.png")
plt.close()

print("📊 2. Generating training vs validation comparison...")

# 2. Train vs Val Performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training vs Validation Performance - Overfitting Analysis', 
             fontsize=15, fontweight='bold')

# Box loss
axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train', linewidth=2)
axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r-', label='Validation', linewidth=2)
axes[0, 0].axvline(x=20, color='green', linestyle=':', alpha=0.7)
axes[0, 0].set_title('Box Localization Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Class loss
axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 'r-', label='Validation', linewidth=2)
axes[0, 1].axvline(x=20, color='green', linestyle=':', alpha=0.7)
axes[0, 1].set_title('Classification Loss', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# DFL loss
axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], 'b-', label='Train', linewidth=2)
axes[1, 0].plot(df['epoch'], df['val/dfl_loss'], 'r-', label='Validation', linewidth=2)
axes[1, 0].axvline(x=20, color='green', linestyle=':', alpha=0.7)
axes[1, 0].set_title('Distribution Focal Loss', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Train-Val gap
gap = abs(df['train/box_loss'] - df['val/box_loss'])
axes[1, 1].plot(df['epoch'], gap, 'purple', linewidth=2.5)
axes[1, 1].axvline(x=20, color='green', linestyle=':', alpha=0.7)
axes[1, 1].fill_between(df['epoch'], 0, gap, alpha=0.3, color='purple')
axes[1, 1].set_title('Train-Val Loss Gap (Overfitting Indicator)', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss Difference')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'train_vs_val_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ train_vs_val_analysis.png")
plt.close()

print("📊 3. Generating model convergence analysis...")

# 3. Convergence Speed
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Convergence Analysis', fontsize=15, fontweight='bold')

# Learning curve
axes[0].plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', linewidth=3, label='Actual mAP')
axes[0].axvline(x=20, color='purple', linestyle=':', label='Stage Transition')
axes[0].axhline(y=0.60, color='orange', linestyle='--', label='Target: 60%', linewidth=2)
axes[0].axhline(y=0.8360, color='red', linestyle='-', label='Achieved: 83.60%', linewidth=2.5)
axes[0].fill_between(df['epoch'], 0.60, df['metrics/mAP50(B)'], 
                     where=(df['metrics/mAP50(B)'] >= 0.60), alpha=0.3, color='green')
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('mAP@0.5', fontweight='bold')
axes[0].set_title('Target Achievement Progress', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Plateau detection
rolling_mean = df['metrics/mAP50(B)'].rolling(window=5).mean()
axes[1].plot(df['epoch'], df['metrics/mAP50(B)'], 'lightblue', alpha=0.5, label='Raw mAP')
axes[1].plot(df['epoch'], rolling_mean, 'b-', linewidth=3, label='5-Epoch Moving Avg')
axes[1].axvline(x=20, color='purple', linestyle=':', label='Stage Transition')
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('mAP@0.5', fontweight='bold')
axes[1].set_title('Learning Smoothness (Plateau Detection)', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ convergence_analysis.png")
plt.close()

print("📊 4. Copying existing 2-class visualizations...")

# Copy existing visualizations
import shutil
viz_to_copy = [
    'figures/training_curves_2class.png',
    'figures/model_comparison.png',
    'figures/per_class_performance_2class.png',
    'figures/performance_dashboard_2class.png',
    'figures/confusion_matrix_2class.png',
    'runs/detect/results/stage2_unfrozen/BoxPR_curve.png',
    'runs/detect/results/stage2_unfrozen/BoxF1_curve.png',
]

for viz in viz_to_copy:
    viz_path = Path(viz)
    if viz_path.exists():
        dest_name = viz_path.name
        shutil.copy(viz_path, output_dir / dest_name)
        print(f"   ✓ {dest_name}")

# Copy model
print("\n🤖 Copying model...")
shutil.copy('models/best.pt', 'research-2-class/model/best_2class.pt')
print("   ✓ best_2class.pt (6.0 MB)")

# Copy training data
print("\n📈 Copying training metrics...")
shutil.copy('runs/detect/results/stage1_frozen/results.csv', 
            'research-2-class/training_data/stage1_results.csv')
shutil.copy('runs/detect/results/stage2_unfrozen/results.csv',
            'research-2-class/training_data/stage2_results.csv')
print("   ✓ stage1_results.csv")
print("   ✓ stage2_results.csv")

print("\n" + "="*80)
print(" ✓ ALL LEARNING ANALYSIS GRAPHS GENERATED")
print("="*80)
print(f"\nLocation: {output_dir.absolute()}")
print("\nGenerated graphs:")
print("  • learning_analysis_comprehensive.png - Complete learning analysis")
print("  • train_vs_val_analysis.png - Overfitting check")
print("  • convergence_analysis.png - Target achievement")
print("  + 7 existing visualizations copied")
print("\n" + "="*80)

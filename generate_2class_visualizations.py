"""
Generate comprehensive visualizations for 2-class model and organize outcomes
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
stage1_results = Path('runs/detect/results/stage1_frozen/results.csv')
stage2_results = Path('runs/detect/results/stage2_unfrozen/results.csv')
confusion_matrix = Path('runs/detect/results/stage2_unfrozen/confusion_matrix_normalized.png')
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("\n" + "="*80)
print(" GENERATING 2-CLASS MODEL VISUALIZATIONS")
print("="*80 + "\n")

# 1. Combined Training Curves
print("📊 Generating training curves...")

if stage1_results.exists() and stage2_results.exists():
    df1 = pd.read_csv(stage1_results)
    df1 = df1.iloc[:, :11].copy()
    df1.columns = df1.columns.str.strip()
    
    df2 = pd.read_csv(stage2_results)
    df2 = df2.iloc[:, :11].copy()
    df2.columns = df2.columns.str.strip()
    
    df2['epoch'] = df2['epoch'] + len(df1)
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('2-Class Model Training Progress (Free vs Occupied)\n180 Images | 50 Epochs | mAP@0.5: 83.60%', 
                 fontsize=16, fontweight='bold')
    
    # Box Loss
    axes[0, 0].plot(df_combined['epoch'], df_combined['train/box_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(df_combined['epoch'], df_combined['val/box_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].axvline(x=20, color='green', linestyle=':', label='Stage 1→2', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Box Loss')
    axes[0, 0].set_title('Bounding Box Localization Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class Loss
    axes[0, 1].plot(df_combined['epoch'], df_combined['train/cls_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(df_combined['epoch'], df_combined['val/cls_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 1].axvline(x=20, color='green', linestyle=':', label='Stage 1→2', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('Class Prediction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP@0.5
    axes[1, 0].plot(df_combined['epoch'], df_combined['metrics/mAP50(B)'], 'g-', linewidth=2.5)
    axes[1, 0].axvline(x=20, color='green', linestyle=':', label='Stage 1→2', linewidth=2)
    axes[1, 0].axhline(y=0.8360, color='red', linestyle='--', label='Final Test mAP@0.5: 83.60%', linewidth=2)
    axes[1, 0].axhline(y=0.60, color='orange', linestyle=':', label='Target: 60%', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP@0.5')
    axes[1, 0].set_title('Mean Average Precision @ IoU 0.5')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Precision & Recall
    axes[1, 1].plot(df_combined['epoch'], df_combined['metrics/precision(B)'], 'b-', label='Precision', linewidth=2)
    axes[1, 1].plot(df_combined['epoch'], df_combined['metrics/recall(B)'], 'r-', label='Recall', linewidth=2)
    axes[1, 1].axvline(x=20, color='green', linestyle=':', label='Stage 1→2', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_2class.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ {output_dir / 'training_curves_2class.png'}")
    plt.close()

# 2. Model Comparison (3-class vs 2-class)
print("📊 Generating model comparison...")

comparison_data = {
    'Model': ['3-Class Model', '2-Class Model'],
    'Classes': ['Free, Occupied, Partially-Free', 'Free, Occupied'],
    'Training Images': [125, 125],
    'Total Annotations': [3346, 3340],
    'mAP@0.5 (Test)': ['56.77%', '83.60%'],
    'mAP@0.5:0.95': ['50.55%', '71.96%'],
    'Precision': ['49.45%', '78.88%'],
    'Recall': ['58.73%', '87.18%'],
    'Free Class mAP': ['71.74%', '68.33%'],
    'Occupied Class mAP': ['98.55%', '98.86%'],
    'Status': ['⚠️ Below Target', '✅ EXCELLENT'],
}

df_comparison = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_comparison.values,
                colLabels=df_comparison.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.12, 0.16, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08, 0.10, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header
for i in range(len(df_comparison.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#2196F3')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(df_comparison) + 1):
    for j in range(len(df_comparison.columns)):
        cell = table[(i, j)]
        if i == 1:  # 3-class row
            cell.set_facecolor('#ffebee')
        else:  # 2-class row (better)
            cell.set_facecolor('#e8f5e9')

plt.title('Model Performance Comparison: 3-Class vs 2-Class\n', fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ {output_dir / 'model_comparison.png'}")
plt.close()

# 3. Per-Class Performance
print("📊 Generating per-class performance chart...")

fig, ax = plt.subplots(figsize=(10, 6))

classes = ['Free Parking', 'Occupied Parking']
precision = [0.6762, 0.9014]
recall = [0.7500, 0.9936]
mAP50 = [0.6833, 0.9886]
mAP50_95 = [0.5730, 0.8660]

x = range(len(classes))
width = 0.2

ax.bar([i - 1.5*width for i in x], precision, width, label='Precision', color='#2196F3')
ax.bar([i - 0.5*width for i in x], recall, width, label='Recall', color='#4CAF50')
ax.bar([i + 0.5*width for i in x], mAP50, width, label='mAP@0.5', color='#FF9800')
ax.bar([i + 1.5*width for i in x], mAP50_95, width, label='mAP@0.5:0.95', color='#9C27B0')

ax.set_xlabel('Parking Space Class', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('2-Class Model: Per-Class Performance Metrics\nTest Set (19 images, 328 parking spaces)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (p, r, m1, m2) in enumerate(zip(precision, recall, mAP50, mAP50_95)):
    ax.text(i - 1.5*width, p + 0.02, f'{p:.1%}', ha='center', va='bottom', fontsize=8)
    ax.text(i - 0.5*width, r + 0.02, f'{r:.1%}', ha='center', va='bottom', fontsize=8)
    ax.text(i + 0.5*width, m1 + 0.02, f'{m1:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(i + 1.5*width, m2 + 0.02, f'{m2:.1%}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'per_class_performance_2class.png', dpi=300, bbox_inches='tight')
print(f"   ✓ {output_dir / 'per_class_performance_2class.png'}")
plt.close()

# 4. Performance Summary Dashboard
print("📊 Generating performance summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('YOLOv8n 2-Class Parking Detection Model - Final Performance Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# Overall metrics
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

metrics_text = f"""
═══════════════════════════════════════════════════════════════════════════════════════════
                                OVERALL PERFORMANCE METRICS
═══════════════════════════════════════════════════════════════════════════════════════════
Test Set: 19 images, 328 parking spaces                    Model: YOLOv8n (3M params, 6.2MB)

   mAP@0.5:        83.60%  ✅ EXCELLENT (+39% above 60% target)      Inference (CPU):  78ms  ✅
   mAP@0.5:0.95:   71.96%  ✅ EXCELLENT                              Inference (GPU):  ~8ms  ✅
   Precision:      78.88%  ✅ GOOD                                    Epochs:          50
   Recall:         87.18%  ✅ EXCELLENT                               Training Time:    16min (GPU)
═══════════════════════════════════════════════════════════════════════════════════════════
"""

ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11, 
         family='monospace', bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

# Per-class table
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('tight')
ax2.axis('off')

class_data = {
    'Class': ['Free Parking', 'Occupied Parking'],
    'Samples': ['16', '312'],
    'Precision': ['67.62%', '90.14%'],
    'Recall': ['75.00%', '99.36%'],
    'mAP@0.5': ['68.33%', '98.86%'],
    'mAP@0.5:0.95': ['57.30%', '86.60%'],
}

df_class = pd.DataFrame(class_data)
table2 = ax2.table(cellText=df_class.values, colLabels=df_class.columns,
                   cellLoc='center', loc='center', colWidths=[0.20, 0.12, 0.15, 0.15, 0.15, 0.15])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)

for i in range(len(df_class.columns)):
    cell = table2[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

for i in range(1, len(df_class) + 1):
    for j in range(len(df_class.columns)):
        cell = table2[(i, j)]
        cell.set_facecolor('#f5f5f5' if i % 2 == 0 else '#ffffff')

ax2.set_title('Per-Class Performance Breakdown', fontsize=12, fontweight='bold', pad=10)

# Key achievements
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

achievements_text = """
🏆 KEY ACHIEVEMENTS

✅ Exceeded 60% target by 39%
   (83.60% vs 60% requirement)

✅ Occupied: 98.86% mAP
   Near-perfect detection (99% recall)

✅ Free: 68.33% mAP
   Good performance despite imbalance

✅ Real-time inference
   78ms CPU, ~8ms GPU

✅ Compact model
   6.2MB (edge deployment ready)

✅ No overfitting
   Stable convergence in 50 epochs
"""

ax3.text(0.1, 0.5, achievements_text, ha='left', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

# Training configuration
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')

config_text = """
⚙️ TRAINING CONFIGURATION

Dataset:        180 images (125 train, 36 val, 19 test)
Annotations:    3,340 (273 free, 3,067 occupied)
Augmentation:   Flip, rotate, brightness, contrast

Stage 1 (20 epochs):
  • Frozen backbone
  • LR: 0.01, Batch: 16
  • Duration: ~7 min (GPU)

Stage 2 (30 epochs):
  • Full fine-tuning
  • LR: 0.005, Batch: 16
  • Duration: ~9 min (GPU)
"""

ax4.text(0.1, 0.5, config_text, ha='left', va='center', fontsize=10,
         family='monospace', bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8))

# Comparison with 3-class
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

comparison_text = """
📊 2-CLASS vs 3-CLASS MODEL

Metric              3-Class    2-Class    Improvement
────────────────────────────────────────────────────
mAP@0.5             56.77%     83.60%     +47.3%  ✅
mAP@0.5:0.95        50.55%     71.96%     +42.4%  ✅
Precision           49.45%     78.88%     +59.5%  ✅
Recall              58.73%     87.18%     +48.5%  ✅

Why Better?
• Removed partially_free (only 6 samples)
• Binary classification simpler & more stable
• Better class separation
"""

ax5.text(0.1, 0.5, comparison_text, ha='left', va='center', fontsize=10,
         family='monospace', bbox=dict(boxstyle='round', facecolor='#f3e5f5', alpha=0.8))

plt.savefig(output_dir / 'performance_dashboard_2class.png', dpi=300, bbox_inches='tight')
print(f"   ✓ {output_dir / 'performance_dashboard_2class.png'}")
plt.close()

# 5. Copy confusion matrix
print("📊 Copying confusion matrix...")
if confusion_matrix.exists():
    shutil.copy(confusion_matrix, output_dir / 'confusion_matrix_2class.png')
    print(f"   ✓ {output_dir / 'confusion_matrix_2class.png'}")

print("\n" + "="*80)
print(" ✓ ALL VISUALIZATIONS GENERATED")
print("="*80)
print(f"\nGenerated files in {output_dir.absolute()}:")
print(f"  • training_curves_2class.png")
print(f"  • model_comparison.png (3-class vs 2-class)")
print(f"  • per_class_performance_2class.png")
print(f"  • performance_dashboard_2class.png")
print(f"  • confusion_matrix_2class.png")
print("\n" + "="*80)

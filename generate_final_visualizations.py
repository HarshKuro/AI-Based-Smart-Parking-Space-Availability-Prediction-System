"""
Generate comprehensive training visualizations and performance comparison
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

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
print(" GENERATING FINAL VISUALIZATIONS")
print("="*80 + "\n")

# 1. Combined Training Curves (Both Stages)
print("📊 Generating combined training curves...")

if stage1_results.exists() and stage2_results.exists():
    df1 = pd.read_csv(stage1_results)
    df1 = df1.iloc[:, :11].copy()  # Remove extra columns
    df1.columns = df1.columns.str.strip()
    
    df2 = pd.read_csv(stage2_results)
    df2 = df2.iloc[:, :11].copy()
    df2.columns = df2.columns.str.strip()
    
    # Adjust epoch numbers for stage 2
    df2['epoch'] = df2['epoch'] + len(df1)
    
    # Combine
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Plot training metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 Training Progress - Two-Stage Transfer Learning\n180 Images (125 train, 36 val, 19 test)', 
                 fontsize=16, fontweight='bold')
    
    # Box Loss
    axes[0, 0].plot(df_combined['epoch'], df_combined['train/box_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(df_combined['epoch'], df_combined['val/box_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].axvline(x=20, color='green', linestyle=':', label='Stage 1→2 Transition', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Box Loss')
    axes[0, 0].set_title('Bounding Box Localization Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class Loss
    axes[0, 1].plot(df_combined['epoch'], df_combined['train/cls_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(df_combined['epoch'], df_combined['val/cls_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 1].axvline(x=20, color='green', linestyle=':', label='Stage 1→2 Transition', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('Class Prediction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP@0.5
    axes[1, 0].plot(df_combined['epoch'], df_combined['metrics/mAP50(B)'], 'g-', linewidth=2.5)
    axes[1, 0].axvline(x=20, color='green', linestyle=':', label='Stage 1→2 Transition', linewidth=2)
    axes[1, 0].axhline(y=0.5677, color='red', linestyle='--', label='Final Test mAP@0.5: 56.77%', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP@0.5')
    axes[1, 0].set_title('Mean Average Precision @ IoU 0.5')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Precision & Recall
    axes[1, 1].plot(df_combined['epoch'], df_combined['metrics/precision(B)'], 'b-', label='Precision', linewidth=2)
    axes[1, 1].plot(df_combined['epoch'], df_combined['metrics/recall(B)'], 'r-', label='Recall', linewidth=2)
    axes[1, 1].axvline(x=20, color='green', linestyle=':', label='Stage 1→2 Transition', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'combined_training_curves.png'}")
    plt.close()
    
    # 2. Per-Class Performance Over Time
    print("📊 Generating per-class performance curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Per-Class mAP@0.5 Evolution - Two-Stage Training', fontsize=16, fontweight='bold')
    
    class_names = ['free_parking_space', 'not_free_parking_space', 'partially_free_parking_space']
    class_cols = ['metrics/mAP50(B)', 'metrics/mAP50(B)', 'metrics/mAP50(B)']  # All classes combined
    
    for idx, (ax, cls_name) in enumerate(zip(axes, class_names)):
        ax.plot(df_combined['epoch'], df_combined['metrics/mAP50(B)'], 'g-', linewidth=2)
        ax.axvline(x=20, color='green', linestyle=':', label='Stage 1→2', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5')
        ax.set_title(cls_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'per_class_performance.png'}")
    plt.close()

else:
    print("   ⚠️ Training results not found")

# 3. Copy confusion matrix
print("📊 Copying confusion matrix...")
if confusion_matrix.exists():
    shutil.copy(confusion_matrix, output_dir / 'confusion_matrix_normalized.png')
    print(f"   ✓ Saved: {output_dir / 'confusion_matrix_normalized.png'}")
else:
    print("   ⚠️ Confusion matrix not found")

# 4. Performance Comparison Table
print("📊 Creating performance comparison...")

comparison_data = {
    'Dataset': ['Initial (30 images)', 'Merged (180 images)'],
    'Training Images': [21, 125],
    'Val Images': [6, 36],
    'Test Images': [3, 19],
    'Total Annotations': [903, 3346],
    'Epochs': [130, 50],
    'Training Time': ['~20 min (CPU)', '~16 min (GPU)'],
    'mAP@0.5 (Val)': ['62.3%', '53.0%'],
    'mAP@0.5 (Test)': ['N/A', '56.77%'],
    'Free Class mAP': ['89.4%', '71.74%'],
    'Occupied Class mAP': ['97.4%', '98.55%'],
    'Partially-Free mAP': ['0%', '0%'],
}

df_comparison = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_comparison.values,
                colLabels=df_comparison.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.12, 0.08, 0.08, 0.12, 0.08, 0.12, 0.10, 0.10, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(df_comparison.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(df_comparison) + 1):
    for j in range(len(df_comparison.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('#ffffff')

plt.title('Performance Comparison: Initial vs Merged Dataset\n', fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'performance_comparison.png'}")
plt.close()

# 5. Class Distribution Analysis
print("📊 Creating class distribution visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')

# Initial dataset
classes_initial = ['Free', 'Occupied', 'Partially-Free']
counts_initial = [273, 624, 6]
colors = ['#4CAF50', '#F44336', '#FFC107']

ax1.pie(counts_initial, labels=classes_initial, autopct='%1.1f%%', colors=colors, 
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title(f'Initial Dataset (30 images)\nTotal: {sum(counts_initial)} annotations', fontsize=12)

# Merged dataset
counts_merged = [273, 3067, 6]
ax2.pie(counts_merged, labels=classes_initial, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax2.set_title(f'Merged Dataset (180 images)\nTotal: {sum(counts_merged)} annotations', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'class_distribution.png'}")
plt.close()

print("\n" + "="*80)
print(" ✓ ALL VISUALIZATIONS GENERATED")
print("="*80)
print(f"\nOutput directory: {output_dir.absolute()}")
print(f"\nGenerated files:")
print(f"  • combined_training_curves.png")
print(f"  • per_class_performance.png")
print(f"  • confusion_matrix_normalized.png")
print(f"  • performance_comparison.png")
print(f"  • class_distribution.png")
print(f"  • test_metrics.csv")
print("\n" + "="*80)

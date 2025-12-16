# CYBERBULLYING DETECTION - BLACK & WHITE FRIENDLY VISUALIZATION
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Set global style for black and white printing with LARGER FONTS
plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Your exact performance data
models = ['SVM', 'Random Forest', 'Logistic Regression']
accuracy = [89.65, 87.48, 76.61]
precision = [81.85, 77.68, 66.67]
recall = [94.50, 95.40, 80.38]
f1_score = [87.72, 85.63, 72.89]

# Your confusion matrix data for SVM (selected model)
confusion_matrix = np.array([
    [2108, 328],  # TN, FP
    [86, 1479]    # FN, TP
])

print("🔄 Generating Black & White Friendly Performance Graphs...")

# =============================================================================
# GRAPH 1: Side-by-Side Performance Comparison (Black & White)
# =============================================================================
plt.figure(figsize=(16, 10))

x = np.arange(len(models))
width = 0.2

# Define patterns and shades for black & white printing
patterns = ['///', '\\\\\\', '|||', '---']
shades = ['white', 'lightgray', 'darkgray', 'black']

# Create bars with different patterns and shades
bars1 = plt.bar(x - 1.5*width, accuracy, width, 
                label='Accuracy', 
                hatch=patterns[0], 
                edgecolor='black',
                facecolor=shades[0],
                linewidth=2.0)

bars2 = plt.bar(x - 0.5*width, precision, width, 
                label='Precision', 
                hatch=patterns[1], 
                edgecolor='black',
                facecolor=shades[1],
                linewidth=2.0)

bars3 = plt.bar(x + 0.5*width, recall, width, 
                label='Recall', 
                hatch=patterns[2], 
                edgecolor='black',
                facecolor=shades[2],
                linewidth=2.0)

bars4 = plt.bar(x + 1.5*width, f1_score, width, 
                label='F1-Score', 
                hatch=patterns[3], 
                edgecolor='black',
                facecolor=shades[3],
                linewidth=2.0)

plt.xlabel('Machine Learning Models', fontsize=16, fontweight='bold')
plt.ylabel('Performance (%)', fontsize=16, fontweight='bold')
plt.title('Model Performance Comparison', fontsize=18, fontweight='bold')
plt.xticks(x, models, fontsize=14, fontweight='bold')
plt.legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=13)
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.ylim(60, 100)

# Add value annotations on bars with LARGER FONT
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.8, 
                f'{height}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('images/performance_comparison_bw.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GRAPH 2: Individual Metrics Subplot (Black & White)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Detailed Performance Analysis', fontsize=20, fontweight='bold', y=0.98)

# Different patterns for each model
model_patterns = ['///', '\\\\\\', 'xxx']
model_shades = ['white', 'lightgray', 'darkgray']

# Plot individual metrics
metrics_data = [
    (accuracy, 'Accuracy', 'Accuracy (%)', (70, 100), axes[0,0]),
    (precision, 'Precision', 'Precision (%)', (60, 100), axes[0,1]),
    (recall, 'Recall', 'Recall (%)', (75, 100), axes[1,0]),
    (f1_score, 'F1-Score', 'F1-Score (%)', (70, 100), axes[1,1])
]

for data, title, ylabel, ylim, ax in metrics_data:
    bars = []
    for i, (model, value) in enumerate(zip(models, data)):
        bar = ax.bar(i, value, 
                    hatch=model_patterns[i],
                    edgecolor='black',
                    facecolor=model_shades[i],
                    linewidth=2.0,
                    alpha=0.9)
        bars.append(bar[0])
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Set straight horizontal labels with LARGER FONT
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=13, fontweight='bold', ha='center')
    
    # Add value labels with white background for readability - LARGER FONT
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='black', alpha=0.9))

# Add legend for model patterns with LARGER FONT (FIXED ERROR)
axes[0,1].legend(bars, models, title="Models", 
                loc='upper right', 
                framealpha=1, 
                edgecolor='black',
                fontsize=13,
                title_fontsize=14)  # FIXED: Removed title_fontproperties

plt.tight_layout()
plt.savefig('images/individual_metrics_bw.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GRAPH 3: Confusion Matrix (Black & White)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create custom confusion matrix plot with grayscale
cax = ax.matshow(confusion_matrix, cmap='Greys', alpha=0.8)

# Add text annotations with contrasting backgrounds - LARGER FONT
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{confusion_matrix[i, j]}', 
                ha='center', va='center', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor='black'))

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Cyberbullying', 'Cyberbullying'], fontsize=14, fontweight='bold')
ax.set_yticklabels(['Non-Cyberbullying', 'Cyberbullying'], fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.xaxis.set_label_position('top')

plt.title('Confusion Matrix - SVM Model', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('images/confusion_matrix_bw.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GRAPH 4: Recall vs Precision Comparison
# =============================================================================
plt.figure(figsize=(12, 8))

# Focus on Recall and Precision for cyberbullying class
x_pos = np.arange(len(models))
width = 0.35

bars_recall = plt.bar(x_pos - width/2, recall, width, 
                      label='Recall', 
                      hatch='///',
                      edgecolor='black',
                      facecolor='white',
                      linewidth=2.0)

bars_precision = plt.bar(x_pos + width/2, precision, width, 
                         label='Precision', 
                         hatch='\\\\\\',
                         edgecolor='black',
                         facecolor='lightgray',
                         linewidth=2.0)

plt.xlabel('Machine Learning Models', fontsize=16, fontweight='bold')
plt.ylabel('Performance (%)', fontsize=16, fontweight='bold')
plt.title('Recall vs Precision Comparison', fontsize=18, fontweight='bold')
plt.xticks(x_pos, models, fontsize=14, fontweight='bold')
plt.legend(framealpha=1, edgecolor='black', fontsize=14)
plt.grid(True, alpha=0.3, axis='y', linewidth=1.0)
plt.ylim(60, 100)

# Add value labels with LARGER FONT
for bars in [bars_recall, bars_precision]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('images/recall_precision_bw.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GRAPH 5: All Metrics Grouped by Model
# =============================================================================
plt.figure(figsize=(14, 9))

# Define metrics data
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Different patterns for models
patterns = ['///', '\\\\\\', 'xxx']
x_pos = np.arange(len(metric_names))  # Position for each metric
width = 0.25

# Plot all metrics for each model
for i, model in enumerate(models):
    model_metrics = [accuracy[i], precision[i], recall[i], f1_score[i]]
    plt.bar(x_pos + i*width, model_metrics, width,
            hatch=patterns[i],
            edgecolor='black',
            facecolor=['white', 'lightgray', 'darkgray'][i],
            linewidth=2.0,
            alpha=0.9,
            label=model)

plt.xlabel('Performance Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Performance (%)', fontsize=16, fontweight='bold')
plt.title('Model Performance Across Metrics', fontsize=18, fontweight='bold')
plt.xticks(x_pos + width, metric_names, fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.ylim(60, 100)

# Add value labels on bars with LARGER FONT
for i, model in enumerate(models):
    model_metrics = [accuracy[i], precision[i], recall[i], f1_score[i]]
    for j, value in enumerate(model_metrics):
        plt.text(x_pos[j] + i*width, value + 0.8, 
                f'{value}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor='black', alpha=0.9))

plt.legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=14)
plt.tight_layout()
plt.savefig('images/model_comparison_bw.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Performance Summary
# =============================================================================
print("\n" + "="*80)
print("CYBERBULLYING DETECTION - PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-"*80)
for i, model in enumerate(models):
    print(f"{model:<20} {accuracy[i]:<10.2f}% {precision[i]:<10.2f}% {recall[i]:<10.2f}% {f1_score[i]:<10.2f}%")
print("-"*80)
print("📊 PATTERNS: '///' = SVM, '\\\\\\' = Random Forest, 'xxx' = Logistic Regression")
print("✅ All graphs optimized for black & white printing with LARGE FONTS")
print("="*80)

print(f"\n✅ SUCCESS! Generated 5 Black & White friendly graphs with LARGE FONTS:")
print(f"   📈 performance_comparison_bw.png - Main comparison")
print(f"   📊 individual_metrics_bw.png - Detailed metrics") 
print(f"   🎯 confusion_matrix_bw.png - SVM confusion matrix")
print(f"   🔍 recall_precision_bw.png - Recall vs Precision focus")
print(f"   📋 model_comparison_bw.png - Grouped metric view")
print(f"\n📍 Location: {os.path.abspath('images')}")
import matplotlib.pyplot as plt

epochs = list(range(15))

mlp_acc = [64.33, 68.48, 69.13, 70.48, 70.92, 72.27, 72.15, 72.08, 72.06, 72.28, 72.70, 71.96, 72.72, 72.90, 71.48]
gelu_acc = [58.35, 68.30, 71.66, 72.76, 74.07, 75.45, 75.80, 76.22, 77.71, 77.84, 78.14, 79.33, 78.74, 80.08, 78.28]
relu_acc = [59.02, 65.50, 72.72, 74.04, 74.42, 77.28, 76.30, 78.13, 78.64, 79.10, 79.03, 79.35, 80.47, 80.29, 81.84]

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(epochs, gelu_acc, 's-', label='ViT + GELU ', linewidth=2, markersize=6, color='#3498db')
ax2.plot(epochs, relu_acc, '^-', label='ViT + ReLU', linewidth=2, markersize=6, color='#2ecc71')

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax2.set_title('Experiment 2: GELU vs ReLU on Setting 1 (A→A)', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)

plt.tight_layout()
plt.savefig('../visualizations/activation_comparison.png', dpi=300, bbox_inches='tight')

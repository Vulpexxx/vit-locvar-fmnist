import os
import matplotlib.pyplot as plt

# Define data extracted from the training logs
epochs = list(range(15))

# --- Validation Accuracy Data ---
vit_gelu_A_A_acc = [58.35, 68.30, 71.66, 72.76, 74.07, 75.45, 75.80, 76.22, 77.71, 77.84, 78.14, 79.33, 78.74, 80.08, 78.28]
vit_relu_A_A_acc = [59.02, 65.50, 72.72, 74.04, 74.42, 77.28, 76.30, 78.13, 78.64, 79.10, 79.03, 79.35, 80.47, 80.29, 81.84]

# Create output directory
output_dir = "../visualizations"
os.makedirs(output_dir, exist_ok=True)

# Initialize Figure
plt.figure(figsize=(10, 6))

# Plot configurations
plt.plot(epochs, vit_gelu_A_A_acc, label='ViT (GELU): Train A / Val A', marker='s', linestyle='-', color='#1f77b4')
plt.plot(epochs, vit_relu_A_A_acc, label='ViT (ReLU): Train A / Val A', marker='^', linestyle='-', color='#ff7f0e')

# Format plot area
plt.title('GELU vs ReLU on Setting 1 (A→A)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# Adjust layout and save the plot
plt.tight_layout()
save_path = os.path.join(output_dir, "activation_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"Comparison curves successfully saved to {save_path}")
plt.close()
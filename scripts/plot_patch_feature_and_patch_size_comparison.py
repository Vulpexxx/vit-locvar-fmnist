import os
import matplotlib.pyplot as plt

# Define data extracted from the training logs
epochs = list(range(15))

# --- Validation Accuracy Data ---
# Linear Patch Methods
linear_patch4_acc  = [53.05, 59.93, 68.51, 69.02, 70.61, 73.21, 73.01, 72.91, 74.74, 75.46, 75.82, 77.73, 77.22, 77.15, 78.23]
linear_patch8_acc  = [61.97, 69.45, 73.26, 75.83, 77.05, 77.57, 78.76, 79.50, 79.34, 79.27, 81.82, 81.29, 81.52, 81.29, 81.13]
linear_patch16_acc = [65.82, 71.67, 70.23, 73.76, 74.06, 74.14, 75.92, 76.28, 77.49, 77.48, 78.62, 78.99, 79.57, 79.49, 79.36]

# Convolutional Patch Methods
conv_patch4_acc  = [51.66, 64.90, 66.12, 69.02, 72.55, 71.35, 71.56, 74.83, 73.73, 75.56, 76.51, 75.03, 77.29, 74.74, 77.48]
conv_patch8_acc  = [62.25, 69.20, 73.99, 74.02, 75.70, 76.21, 78.44, 78.95, 79.40, 79.50, 79.60, 79.39, 81.41, 80.34, 80.37]
conv_patch16_acc = [66.36, 71.13, 70.95, 74.37, 73.87, 75.73, 75.87, 75.86, 77.80, 77.66, 77.10, 78.33, 77.65, 78.92, 78.11]

# Create output directory
output_dir = "../visualizations"
os.makedirs(output_dir, exist_ok=True)

# Initialize Figure
plt.figure(figsize=(12, 6))

# Plot configurations
# Using solid lines for Linear and dashed lines for Conv for better visual distinction
plt.plot(epochs, linear_patch4_acc,  label='Linear-Patch4',  marker='o', linestyle='-',  color='#1f77b4')
plt.plot(epochs, linear_patch8_acc,  label='Linear-Patch8',  marker='s', linestyle='-',  color='#ff7f0e')
plt.plot(epochs, linear_patch16_acc, label='Linear-Patch16', marker='^', linestyle='-',  color='#2ca02c')

plt.plot(epochs, conv_patch4_acc,    label='Conv-Patch4',    marker='d', linestyle='--', color='#d62728')
plt.plot(epochs, conv_patch8_acc,    label='Conv-Patch8',    marker='v', linestyle='--', color='#9467bd')
plt.plot(epochs, conv_patch16_acc,   label='Conv-Patch16',   marker='p', linestyle='--', color='#8c564b')

# Format plot area
plt.title('Validation Accuracy Convergence Curve of Different Patch Methods and Patch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# Adjust layout and save the plot
plt.tight_layout()
save_path = os.path.join(output_dir, "patch_feature_and_patch_size_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"Comparison curves successfully saved to {save_path}")
plt.close()
import os
import matplotlib.pyplot as plt

# Define data extracted from the training logs
epochs = list(range(15))

# --- ViT Validation Accuracy Data ---
vit_A_A_acc  = [58.350, 68.300, 71.660, 72.760, 74.070, 75.450, 75.800, 76.200, 77.710, 77.840, 78.140, 79.330, 78.740, 80.080, 78.280]
vit_B_B_acc  = [81.220, 84.200, 84.300, 84.940, 86.520, 86.580, 86.610, 86.020, 86.940, 86.710, 87.010, 87.470, 87.410, 88.390, 87.850]
vit_A_B_acc  = [62.840, 68.500, 70.930, 74.210, 77.760, 77.550, 77.690, 79.370, 79.150, 79.880, 80.270, 80.610, 80.760, 79.820, 80.540]
vit_B_A_acc  = [17.900, 17.700, 18.400, 16.890, 18.650, 19.190, 18.960, 18.300, 16.840, 16.970, 15.870, 18.370, 19.900, 18.520, 19.730]

# --- CNN Validation Accuracy Data ---
cnn_A_A_acc  = [82.430, 85.380, 84.190, 87.520, 88.530, 88.440, 87.370, 88.770, 88.920, 89.340, 89.660, 88.980, 89.200, 90.140, 89.800]
cnn_B_A_acc  = [28.580, 28.860, 30.720, 30.090, 29.370, 28.910, 35.910, 32.580, 33.930, 35.490, 31.450, 32.760, 31.630, 33.390, 34.390]
cnn_B_B_acc  = [88.010, 89.970, 90.480, 90.570, 91.460, 91.510, 91.760, 91.530, 92.130, 91.090, 91.300, 90.720, 92.540, 92.270, 92.150] 
cnn_A_B_acc  = [80.720, 86.660, 87.120, 88.910, 88.470, 88.940, 89.990, 89.720, 90.330, 89.100, 89.510, 89.280, 87.140, 88.070, 87.710] 

# Create output directory
output_dir = "../visualizations"
os.makedirs(output_dir, exist_ok=True)

# Initialize Figure with side-by-side subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot ViT Accuracy curves on the left subplot (ax1)
ax1.plot(epochs, vit_A_A_acc, label='Train A / Val A', marker='o', color='#1f77b4')
ax1.plot(epochs, vit_B_B_acc, label='Train B / Val B', marker='s', color='#2ca02c')
ax1.plot(epochs, vit_A_B_acc, label='Train A / Val B', marker='^', color='#d62728')
ax1.plot(epochs, vit_B_A_acc, label='Train B / Val A', marker='d', color='#ff7f0e')
ax1.set_title('ViT Validation Accuracy Curves')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot CNN Accuracy curves on the right subplot (ax2)
ax2.plot(epochs, cnn_A_A_acc, label='Train A / Val A', marker='o', color='#1f77b4')
ax2.plot(epochs, cnn_B_B_acc, label='Train B / Val B', marker='s', color='#2ca02c')
ax2.plot(epochs, cnn_A_B_acc, label='Train A / Val B', marker='^', color='#d62728')
ax2.plot(epochs, cnn_B_A_acc, label='Train B / Val A', marker='d', color='#ff7f0e')
ax2.set_title('CNN Validation Accuracy Curves')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout and save the plot
plt.tight_layout()
save_path = os.path.join(output_dir, "cnn_and_vit_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"Subplot performance curves successfully saved to {save_path}")
plt.close()
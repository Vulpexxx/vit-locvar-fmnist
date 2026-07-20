import os
import matplotlib.pyplot as plt

# Define data extracted from the training logs
epochs = list(range(15))

# --- ViT Validation Accuracy Data ---
vit_A_A_acc  = [58.350, 68.300, 71.660, 72.760, 74.070, 75.450, 75.800, 76.200, 77.710, 77.840, 78.140, 79.330, 78.740, 80.080, 78.280]
vit_B_B_acc  = [81.220, 84.200, 84.300, 84.940, 86.520, 86.580, 86.610, 86.020, 86.940, 86.710, 87.010, 87.470, 87.410, 88.390, 87.850]
vit_A_B_acc  = [62.840, 68.500, 70.930, 74.210, 77.760, 77.550, 77.690, 79.370, 79.150, 79.880, 80.270, 80.610, 80.760, 79.820, 80.540]
vit_B_A_acc  = [17.900, 17.700, 18.400, 16.890, 18.650, 19.190, 18.960, 18.300, 16.840, 16.970, 15.870, 18.370, 19.900, 18.520, 19.730]

# --- MLP Validation Accuracy Data ---
mlp_A_A_acc  = [65.620, 67.090, 70.430, 70.100, 71.050, 71.210, 71.710, 72.000, 72.600, 72.300, 72.290, 73.010, 73.510, 72.110, 73.230]
mlp_B_A_acc  = [13.130, 13.550, 13.900, 14.210, 13.820, 13.710, 14.050, 14.080, 13.950, 14.090, 13.720, 13.610, 14.110, 13.480, 13.850]
mlp_B_B_acc  = [84.520, 86.340, 85.610, 87.360, 88.010, 87.790, 88.110, 88.960, 87.750, 88.000, 88.400, 89.280, 89.230, 88.460, 88.810] 
mlp_A_B_acc  = [67.120, 69.360, 69.110, 70.480, 72.490, 66.530, 69.270, 71.160, 70.730, 73.020, 73.670, 72.200, 71.650, 72.940, 73.570] 

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

# Plot MLP Accuracy curves on the right subplot (ax2)
ax2.plot(epochs, mlp_A_A_acc, label='Train A / Val A', marker='o', color='#1f77b4')
ax2.plot(epochs, mlp_B_B_acc, label='Train B / Val B', marker='s', color='#2ca02c')
ax2.plot(epochs, mlp_A_B_acc, label='Train A / Val B', marker='^', color='#d62728')
ax2.plot(epochs, mlp_B_A_acc, label='Train B / Val A', marker='d', color='#ff7f0e')
ax2.set_title('MLP Validation Accuracy Curves')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout and save the plot
plt.tight_layout()
save_path = os.path.join(output_dir, "mlp_and_vit_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"Subplot performance curves successfully saved to {save_path}")
plt.close()
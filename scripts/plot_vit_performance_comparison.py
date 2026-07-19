import os
import matplotlib.pyplot as plt

# Define data extracted from the training logs
epochs = list(range(15))

# Scenario 1: Train on A, Validate on A (vit_A_A)
vit_A_A_loss = [1.0733, 0.8545, 0.7651, 0.7432, 0.6970, 0.6588, 0.6623, 0.6541, 0.6053, 0.6044, 0.6093, 0.5727, 0.5831, 0.5543, 0.6132]
vit_A_A_acc  = [58.350, 68.300, 71.660, 72.760, 74.070, 75.450, 75.800, 76.200, 77.710, 77.840, 78.140, 79.330, 78.740, 80.080, 78.280]

# Scenario 2: Train on B, Validate on B (vit_B_B)
vit_B_B_loss = [0.5132, 0.4336, 0.4262, 0.4104, 0.3720, 0.3767, 0.3747, 0.3898, 0.3582, 0.3693, 0.3584, 0.3503, 0.3460, 0.3354, 0.3370]
vit_B_B_acc  = [81.220, 84.200, 84.300, 84.940, 86.520, 86.580, 86.610, 86.020, 86.940, 86.710, 87.010, 87.470, 87.410, 88.390, 87.850]

# Scenario 3: Train on A, Validate on B (vit_A_B)
vit_A_B_loss = [1.0329, 0.8488, 0.7998, 0.7097, 0.6233, 0.6218, 0.6230, 0.5711, 0.5892, 0.5414, 0.5428, 0.5391, 0.5306, 0.5612, 0.5485]
vit_A_B_acc  = [62.840, 68.500, 70.930, 74.210, 77.760, 77.550, 77.690, 79.370, 79.150, 79.880, 80.270, 80.610, 80.760, 79.820, 80.540]

# Scenario 4: Train on B, Validate on A (vit_B_A)
vit_B_A_loss = [4.7553, 4.8635, 4.8363, 5.2661, 5.2897, 5.4974, 5.0768, 5.6096, 5.3498, 5.2465, 5.1367, 5.5847, 5.8768, 5.8597, 5.4494]
vit_B_A_acc  = [17.900, 17.700, 18.400, 16.890, 18.650, 19.190, 18.960, 18.300, 16.840, 16.970, 15.870, 18.370, 19.900, 18.520, 19.730]

# Create output directory
output_dir = "../visualizations"
os.makedirs(output_dir, exist_ok=True)

# Initialize Figure with side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Validation Loss curves
ax1.plot(epochs, vit_A_A_loss, label='Train A / Val A', marker='o', color='#1f77b4')
ax1.plot(epochs, vit_B_B_loss, label='Train B / Val B', marker='s', color='#2ca02c')
ax1.plot(epochs, vit_A_B_loss, label='Train A / Val B', marker='^', color='#d62728')
ax1.plot(epochs, vit_B_A_loss, label='Train B / Val A', marker='d', color='#ff7f0e')
ax1.set_title('ViT Validation Loss Curves')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot Validation Accuracy curves
ax2.plot(epochs, vit_A_A_acc, label='Train A / Val A', marker='o', color='#1f77b4')
ax2.plot(epochs, vit_B_B_acc, label='Train B / Val B', marker='s', color='#2ca02c')
ax2.plot(epochs, vit_A_B_acc, label='Train A / Val B', marker='^', color='#d62728')
ax2.plot(epochs, vit_B_A_acc, label='Train B / Val A', marker='d', color='#ff7f0e')
ax2.set_title('ViT Validation Accuracy Curves')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
save_path = os.path.join(output_dir, "vit_performance_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"ViT performance curves successfully saved to {save_path}")
plt.close()
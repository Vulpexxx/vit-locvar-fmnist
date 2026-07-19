import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the root directory to sys.path to import custom modules
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.vit import ViT
from datasets.custom_fmnist import CustomFashionMNIST

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Attention Weights of ViT Model')

    parser.add_argument('--model-path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--data-path', type=str, help='Path to the dataset')
    parser.add_argument('--img-size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--patch-size', type=int, default=8, help='Size of the image patches')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default='../visualizations', help='Directory to save the visualizations')

    args = parser.parse_args()
    return args

def visualize_attention(model_path='../checkpoints/vit_A_A/vit_A_A.pth', data_path='../processed_data/A_fmnist_test.pt', img_size=64, patch_size=8, num_samples=4, save_dir='../visualizations'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = CustomFashionMNIST(data_path)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # Initialize and load model weights
    model = ViT(img_size=img_size, patch_size=patch_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Forward pass to extract attention weights
    with torch.no_grad():
        logits, attn_weights = model.forward_with_attn(images)
    
    # attn_weights: List of length 'depth'. Element shape: [B, num_heads, seq_len, seq_len]
    # seq_len = num_patches + 1 (including CLS token)
    num_patches = (img_size // patch_size) ** 2
    
    # Extract attention weights from the first block (index 0)
    attn = attn_weights[0]  # Shape: [B, num_heads, seq_len, seq_len]
    
    # Average across all heads for the first sample, extracting CLS token attention (excluding self-attention)
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # Shape: [num_patches]
    cls_attn = cls_attn.cpu().numpy().reshape(num_patches, 1)
    
    # Upsample attention weights to original image resolution
    heatmap = np.zeros((img_size, img_size))
    patch_size_px = patch_size
    
    for i in range(num_patches):
        row = i // (img_size // patch_size_px)
        col = i % (img_size // patch_size_px)
        heatmap[row * patch_size_px:(row + 1) * patch_size_px, col * patch_size_px:(col + 1) * patch_size_px] = cls_attn[i]
    
    # Visualize original image, attention heatmap, and spatial overlay
    img = images[0].cpu().squeeze().numpy()  # Shape: [64, 64]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Attention Heatmap (CLS)')
    axes[1].axis('off')
    
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(heatmap, cmap='hot', alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'attention_visualization_{model_path.split("/")[-1][4:7]}.png')
    plt.savefig(save_path, bbox_inches='tight')

    print(f"Attention visualization saved to {save_path}")


if __name__ == '__main__':
    args = parse_args()
    visualize_attention(
        model_path=args.model_path,
        data_path=args.data_path,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
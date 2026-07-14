import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
import torchvision.transforms as transform

def create_custom_fashion_mnist(root_dir='./data', save_dir='./processed_data', canvas_size=64, is_train=True, mode='translated'):
  """
  Generates a custom FashionMNIST dataset and saves it as a .pt file.
  
  Args:
    root_dir: Directory to download the original data.
    save_dir: Directory to save the processed .pt tensors.
    canvas_size: The size of the background canvas (e.g., 128x128).
    is_train: Boolean indicating whether to process train or test set.
    mode: 'translated' (random position) or 'centered' (fixed center position).
  """
  print(f"Preparing {mode} {'train' if is_train else 'test'} dataset...")
  
  # Load original FashionMNIST
  dataset = torchvision.datasets.FashionMNIST(
    root=root_dir, 
    train=is_train, 
    download=True, 
    transform=transform.ToTensor()
  )

  num_samples = len(dataset)
  img_size = 28
  
  # Initialize empty tensors for images, labels, and bounding box positions
  processed_images = torch.zeros((num_samples, 1, canvas_size, canvas_size), dtype=torch.float32)
  labels = torch.zeros(num_samples, dtype=torch.long)
  positions = torch.zeros((num_samples, 2), dtype=torch.long)

  max_y = canvas_size - img_size
  max_x = canvas_size - img_size

  for i in range(num_samples):
    img, label = dataset[i]
    
    # Determine the position based on the selected mode
    if mode == 'translated':
      y = np.random.randint(0, max_y + 1)
      x = np.random.randint(0, max_x + 1)
    elif mode == 'centered':
      y = max_y // 2
      x = max_x // 2
    else:
      raise ValueError("Mode must be either 'translated' or 'centered'")

    # Place the original image onto the canvas
    processed_images[i, 0, y:y+img_size, x:x+img_size] = img[0]
    labels[i] = label
    positions[i] = torch.tensor([y, x])

  # Create directory and save the dataset
  os.makedirs(save_dir, exist_ok=True)
  filename = f"{mode}_fashion_mnist_{'train' if is_train else 'test'}.pt"
  save_path = os.path.join(save_dir, filename)

  torch.save({
    'images': processed_images,
    'labels': labels,
    'positions': positions
  }, save_path)
  
  print(f"=> Successfully saved to: {save_path}")

  return processed_images, labels, positions

def visualize_samples(images, labels, positions, num_samples=5, save_name="samples.png"):
  """Visualizes the processed samples with bounding boxes."""
  fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
  if num_samples == 1:
    axes = [axes]

  classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  for i in range(num_samples):
    img = images[i][0].numpy()
    y, x = positions[i].numpy()
    label_idx = labels[i].item()
    
    ax = axes[i]
    ax.imshow(img, cmap='gray')
    
    rect = patches.Rectangle((x, y), 28, 28, linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Label: {classes[label_idx]}\nPos: (y={y}, x={x})", fontsize=10)
    ax.axis('off')
      
  plt.tight_layout()
  
  save_dir = "./visualizations"
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, save_name)
  plt.savefig(save_path, bbox_inches='tight', dpi=150)
  print(f"=> Visualization saved to: {save_path}")

  plt.close(fig)


if __name__ == "__main__":
  # Set random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  # 1. Generate TRANSLATED datasets
  train_imgs_t, train_labels_t, train_pos_t = create_custom_fashion_mnist(is_train=True, mode='translated')
  test_imgs_t, test_labels_t, test_pos_t = create_custom_fashion_mnist(is_train=False, mode='translated')

  # 2. Generate CENTERED datasets
  train_imgs_c, train_labels_c, train_pos_c = create_custom_fashion_mnist(is_train=True, mode='centered')
  test_imgs_c, test_labels_c, test_pos_c = create_custom_fashion_mnist(is_train=False, mode='centered')

  # 3. Visualize to confirm
  print("Visualizing Translated Samples...")
  visualize_samples(train_imgs_t, train_labels_t, train_pos_t, num_samples=5, save_name="translated_samples.png")
  print("Visualizing Centered Samples...")
  visualize_samples(train_imgs_c, train_labels_c, train_pos_c, num_samples=5, save_name="centered_samples.png")
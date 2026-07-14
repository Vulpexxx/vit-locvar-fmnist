import torch
import torch.nn as nn

class CNN(nn.Module):
  """
  A simple Convolutional Neural Network baseline.
  Uses AdaptiveMaxPool2d to automatically handle different canvas sizes (e.g., 64 or 128).
  """
  def __init__(self, in_chans=1, num_classes=10):
    super(CNN, self).__init__()
    
    # Feature extraction with Convolutional blocks
    self.features = nn.Sequential(
      # Block 1
      nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      # Block 2
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      # Block 3
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    # Global Max Pooling ensures the output is always 2x2 spatially
    # regardless of whether the input is 64x64 or 128x128
    self.pool = nn.AdaptiveMaxPool2d((2, 2))
    
    # Classification Head
    self.classifier = nn.Sequential(
      nn.Linear(128*2*2, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, num_classes)
    )

  def forward(self, x):
    # Extract spatial features
    x = self.features(x)
    # Pool features to a fixed size
    x = self.pool(x)
    # Flatten the tensor for the linear layer: [B, 128, 2, 2] -> [B, 128*2*2]
    x = torch.flatten(x, 1)
    # Output class logits
    out = self.classifier(x)
    return out
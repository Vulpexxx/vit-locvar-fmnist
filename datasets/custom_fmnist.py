import torch
from torch.utils.data import Dataset

class CustomFashionMNIST(Dataset):
  def __init__(self, data_path):
    data = torch.load(data_path)
    self.images = data['images']
    self.labels = data['labels']
    self.positions = data['positions']

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img = self.images[idx]
    label = self.labels[idx]
    return img, label

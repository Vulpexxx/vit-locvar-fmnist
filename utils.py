import torch
from typing import Tuple, List

class AverageMeter(object):
  '''Compute and store the average and current value of a metric (e.g., loss or accuracy)'''
  def __init__(self):
    self.reset()

  def reset(self):
    '''Resets all statistics to zero'''
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0

  def update(self, val, n = 1):
    '''Updates the statistics with a new value
    :param val: The new value to incorporate
    :param n: The number of samples in the current batch
    '''
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
  '''Computes the accuracy over the k top predictions for the specified values of k
  :param output: The model's output logits, shape [batch_size, num_classes]
  :param target: The ground truth labels, shape [batch_size]
  :param topk: A tuple specifying the values of k for which to compute accuracy (e.g., (1,) or (1, 5))
  :return: A list containing the accuracy percentages for each specified k
  '''
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    # Get the indices of the top k predictions for each sample
    # output.topk returns both the values and indices; we only need the indices
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)

    # Transpose pred to have shape [maxk, batch_size] for easier comparison with target
    pred = pred.t()

    # Compare the top k predictions with the ground truth labels
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      # Count the number of correct predictions in the top k
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

      # Compute the accuracy as a percentage
      res.append(correct_k.mul_(100.0 / batch_size))

    return res
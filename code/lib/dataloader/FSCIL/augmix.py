from .augmentations import *
import numpy as np
from PIL import Image
import torch

def aug(image,preprocess, width=3,depth=-1):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations_all




  # if args.all_ops:
  #   aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(np.array(image)))
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, 3)
    # Preprocessing commutes since all coefficients are convex

    mix += ws[i] * preprocess(image_aug)


  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed



from torchvision import transforms

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd


    self.random_f =transforms.RandomHorizontalFlip()

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:

      # return  self.preprocess(x), aug(x, self.preprocess), y

      return self.preprocess(x), self.preprocess(self.random_f(x)), y
    else:
      # im_tuple = (self.preprocess(x), aug(x, self.preprocess),
      #             aug(x, self.preprocess))
      # return im_tuple, y
      return self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess), y

  def __len__(self):
    return len(self.dataset)




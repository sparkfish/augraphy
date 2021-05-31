import random
import numpy as np

from Augraphy.ImageTransformer import ImageTransformer

class InkBleedAugmentation(ImageTransformer):
  def __init__(self, intensity, debug=False):
    super().__init__(debug=debug)
    self.intensity = intensity
    add_noise = lambda x, y: random.randint(0, 224) if (y == 255 and random.random() < self.intensity) else x
    self.add_noise = np.vectorize(add_noise)

  def __call__(self, img):
    sobel = self.transform(self.sobel, img)
    return self.add_noise(img, sobel)
  
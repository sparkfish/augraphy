from sklearn.datasets import make_blobs
from glob import glob

import random
import numpy as np
import cv2

from Augraphy.ImageTransformer import ImageTransformer

class LowInkAugmentation(ImageTransformer):
  def __init__(self, use_periodic_lines=True, use_consistent_lines=True, debug=False):
    super().__init__(debug=debug)
    self.use_periodic_lines = use_periodic_lines
    self.use_consistent_lines = use_consistent_lines

    inconsistent_transparency_line = lambda x: random.randint(0, 255)
    self.inconsistent_transparency_line = np.vectorize(inconsistent_transparency_line)

  def create_blob(self, size=10, points=30, std_min=10, std_max=30):
    std = random.randint(std_min, std_max) / 100
    X, y = make_blobs(n_samples=points, cluster_std=[std], centers=[(0, 0)], n_features=20) #, random_state=1)
    X *= (size // 4)
    X += (size // 2)
    X = [[int(item) for item in items] for items in X]
    blob = np.full((size, size, 1), 0, dtype="uint8")

    for point in X:
      if point[0] < blob.shape[0] and point[1] < blob.shape[1] and point[0] > 0 and point[1] > 0:
        value = random.randint(180,250)
        blob[point[0], point[1]] = value
        
    return blob

  def apply_blob(self, mask):
    size = random.randint(10, 20)
    points = random.randint(5, 25)
    blob = self.create_blob(size, points, 10, 75)

    x_start = random.randint(0, mask.shape[1] - blob.shape[1])
    y_start = random.randint(0, mask.shape[0] - blob.shape[0])
    x_stop = x_start + blob.shape[1]
    y_stop = y_start + blob.shape[0]
    mask[y_start:y_stop, x_start:x_stop] = blob[:,:,0]

  def add_transparency_line(self, mask, y, alpha=None):
    if (alpha == None):
      alpha = random.randint(16, 224)

    if (self.use_consistent_lines):
      mask[y, :] = np.full(mask[y, :].shape, alpha, dtype="uint8")
    else:
      mask[y, :] = self.inconsistent_transparency_line(mask[y, :])

  def add_random_transparency_lines(self, mask, line_count):
    for i in range(line_count):
      self.add_transparency_line(mask, random.randint(1, mask.shape[0]-1))

  def add_periodic_transparency_line(self, mask, line_count, offset, alpha):
    period = mask.shape[0] // line_count
    
    for y in range(mask.shape[0]-offset):
      if (y % period == 0):
        self.add_transparency_line(mask, y+offset, alpha)

  def add_periodic_transparency_lines(self, mask, lines, line_periods):
    period = mask.shape[0] // line_periods
    self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, 5), alpha=random.randint(96, 255))

    for i in range(lines):
      self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, period), alpha=random.randint(16, 96))

  def create_low_ink_mask(self, image, add_spots=True, add_lines=True, spot_count=30, lines=5, line_periods=10):
    mask = np.full((image.shape[0], image.shape[1]), 0, dtype="uint8")

    for i in range(spot_count):
      self.apply_blob(mask)

    if (self.use_periodic_lines):
      self.add_periodic_transparency_lines(mask, lines, line_periods)
    else:
      self.add_random_transparency_lines(mask, lines*3)
      
    mask = cv2.GaussianBlur(mask,(3,3),0)
    return mask

  def __call__(self, image, add_spots=True, add_lines=True, spot_count=30, line_count=10):
    rotate = random.choice([True, False])

    if (not self.debug and rotate):
      image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    mask = self.transform(self.create_low_ink_mask, image, add_spots, add_lines, spot_count, line_count)
    transparency_lambda = lambda x, y: int((255-y)/255.0 * x) #~y if x != 0 and y != 0 else x
    update_transparency = np.vectorize(transparency_lambda)

    image[:,:,3] = update_transparency(image[:,:,3], mask)

    if (not self.debug and rotate):
      image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image
  
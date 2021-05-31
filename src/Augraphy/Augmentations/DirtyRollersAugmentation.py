import random
import numpy as np
import cv2

from Augraphy.ImageTransformer import ImageTransformer

class DirtyRollersAugmentation(ImageTransformer):
  def __init__(self, line_width=None, debug=False):
    super().__init__(debug=debug)
    if (line_width is None):
      self.line_width = random.randint(8, 12)
    else:
      self.line_width=line_width

  def apply_scanline_mask(self, img, mask, meta_mask):
    if (random.choice([True, False])):
      return self.apply_scanline_mask_v2(img, mask, meta_mask)
    else:
      return self.apply_scanline_mask_v1(img, mask, meta_mask)

  def apply_scanline_mask_v2(self, img, mask, meta_mask):
    mask = self.transform(self.apply_scanline_metamask_v2, mask, meta_mask)
    update_lambda = lambda x, y: min(255, x + (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_metamask_v2(self, img, mask):
    update_lambda = lambda x, y: max(0, x - (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_mask_v1(self, img, mask, meta_mask):
    mask = self.transform(self.apply_scanline_metamask_v1, mask, meta_mask)
    update_lambda = lambda x, y: max(0, x - (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_metamask_v1(self, img, mask):
    update_lambda = lambda x, y: min(99, x + (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def __call__(self, image):
    rotate = random.choice([True, False])

    if (not self.debug and rotate):
      image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = self.transform(self.create_scanline_mask, image.shape[1], image.shape[0], self.line_width)
    meta_mask = self.transform(self.create_scanline_mask, image.shape[1], image.shape[0], self.line_width * random.randint(10, 25))
    image = self.transform(self.apply_scanline_mask, image, mask, meta_mask).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if (not self.debug and rotate):
      image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

  def create_scanline_mask(self, width, height, line_width):
    grad_list = list()

    # Create Standard Bar
    grad_high_pct = random.randint(86, 99)
    grad_low_pct = random.randint(70,85)

    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Wide Dark
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_center = np.full((random.randint(1, 6)), grad_low_pct)
    grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Wide Light
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)
    
    # Create Standard Bar with Lower Dark
    grad_high_pct += min(100, random.randint(-3, 3))
    grad_low_pct -= random.randint(5, 8)
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Low Dark and Wide Dark
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_center = np.full((random.randint(1, 6)), grad_low_pct)
    grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Low Dark Wide Light
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    mask = random.choice(grad_list)
    while (mask.shape[1] < width):
      mask = np.hstack((mask, random.choice(grad_list)))

    return mask[:,0:width]
  
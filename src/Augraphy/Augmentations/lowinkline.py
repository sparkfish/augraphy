################################################################################
# File: lowinkline.py
#
# Class: LowInkLineAugmentation
#
# Description: This file contains a class defining an Augmentation which other
#              Augmentations can use to define streak behavior common in
#              printers running out of ink.
################################################################################



################################################################################
# Imports
################################################################################

import numpy as np
import random



################################################################################
# Class Definitions
################################################################################

class LowInkLineAugmentation(Augmentation):
  def __init__(self, use_consistent_lines=True, probability=0.5):
    super().__init__(probability=probability)

    self.use_consistent_lines = use_consistent_lines
    inconsistent_transparency_line = lambda x: random.randint(0, 255)
    self.inconsistent_transparency_line = np.vectorize(inconsistent_transparency_line)

    apply_line = lambda x, y: x if x > y else y
    self.apply_line = np.vectorize(apply_line)

  # Constructs a string representation of this Augmentation.
  def __repr__(self):
    return f"LowInkLineAugmentation(use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

  # Takes an image, a vertical position, and an opacity value,
  # then adds a line at that position in the image with the given
  # opacity.
  def add_transparency_line(self, mask, y, alpha=None):
    if (alpha == None):
      alpha = random.randint(16, 224)

    if (self.use_consistent_lines):
      low_ink_line = np.full(mask[y, :].shape, alpha, dtype="uint8")
    else:
      low_ink_line = self.inconsistent_transparency_line(mask[y, :])

    mask[y, :] = self.apply_line(mask[y, :], low_ink_line)

    return mask

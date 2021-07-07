################################################################################
# File: lowinkrandomlines.py
#
# Class: LowInkRandomLinesAugmentation
#
# Description: This file contains a class defining an Augmentation which adds
#              low ink lines randomly throughout the image.
################################################################################



################################################################################
# Imports
################################################################################

import random

from Augraphy.Augmentations import AugmentationResult
from Augraphy.augmentations.lowinkline import LowInkLineAugmentation




################################################################################
# Class Definition
################################################################################

class LowInkRandomLinesAugmentation(LowInkLineAugmentation):
  def __init__(self, count_range=(5, 10), use_consistent_lines=True, probability=0.5):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability)
    self.count_range = count_range

  # Constructs a string representation of this Augmentation.
  def __repr__(self):
    return f"LowInkRandomLinesAugmentation(count_range={self.count_range}, use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

  # Applies the Augmentation to input data.
  def __call__(self, data, force=False):
    if (force or self.should_run()):
      mask = data['ink'][-1].result.copy()
      count = random.randint(self.count_range[0], self.count_range[1])

      for i in range(count):
        mask = self.add_transparency_line(mask, random.randint(1, mask.shape[0]-1))

      data['ink'].append(AugmentationResult(self, mask))

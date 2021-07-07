################################################################################
# File: lowinkperiodiclines.py
#
# Class: LowInkPeriodicLinesAugmentation
#
# Description: This file contains a class defining an Augmentation which creates
#              a set of lines that repeat in a periodic fashion throughout the
#              image.
################################################################################



################################################################################
# Imports
################################################################################

import random

import Augraphy.Augmentations
from Augraphy.augmentations.lowinkline import LowInkLineAugmentation



################################################################################
# Class Definition
################################################################################

class LowInkPeriodicLinesAugmentation(LowInkLineAugmentation):
  def __init__(self, count_range=(2, 5), period_range=(10, 30), use_consistent_lines=True, probability=0.5):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability)
    self.count_range = count_range
    self.period_range = period_range

  # Constructs a string representation of this Augmentation.
  def __repr__(self):
    return f"LowInkPeriodicLinesAugmentation(count_range={self.count_range}, period_range={self.period_range}, use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

  # Takes an image, a number of lines to print, an offset value, and
  # an opacity value, then creates horizontal lines of the given opacity
  # over the input image at the specified y-positions, starting at the offset.
  def add_periodic_transparency_line(self, mask, line_count, offset, alpha):
    period = mask.shape[0] // line_count

    for y in range(mask.shape[0]-offset):
      if (y % period == 0):
        self.add_transparency_line(mask, y+offset, alpha)

  # Takes an image, a number of lines to print, and a distance between them,
  # then creates horizontal lines of random opacity over the input image
  # at random intervals.
  def add_periodic_transparency_lines(self, mask, lines, line_periods):
    period = mask.shape[0] // line_periods
    self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, 5), alpha=random.randint(96, 255))

    for i in range(lines):
      self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, period), alpha=random.randint(16, 96))

  # Applies the Augmentation to input data.
  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['ink'][-1].result.copy()
      count = random.randint(self.count_range[0], self.count_range[1])
      period = random.randint(self.period_range[0], self.period_range[1])

      for i in range(count):
        self.add_periodic_transparency_lines(image, count, period)

      data['ink'].append(AugmentationResult(self, image))

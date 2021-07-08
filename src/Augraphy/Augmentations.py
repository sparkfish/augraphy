################################################################################
# File: Augmentations.py
#
# Classes: Augmentation
#          AugmentationResult
#          AugmentationSequence(Augmentation)
#          OneOf(Augmentation)
#
# Description: This file contains classes defining the base Augmentation,
#              a sequence of Augmentations, a randomly selected Augmentation,
#              and the result of an application of an Augmentation.
################################################################################


################################################################################
# Imports
################################################################################

import cv2
import numpy as np
import random

from glob import glob
from scipy.stats import norm
from sklearn.datasets import make_blobs


################################################################################
# Class Definitions
################################################################################

# The base Augmentation class which everything else inherits.
class Augmentation:
    def __init__(self, probability=0.5):
        self.probability = probability

    # Determines whether or not the augmentation should be
    # applied by callers.
    def should_run(self):
        return random.uniform(0.0, 1.0) <= self.probability


# Contains the result of an Augmentation's application.
class AugmentationResult:
    def __init__(self, augmentation, result, metadata=None):
        self.augmentation = augmentation
        self.result = result
        self.metadata = metadata



# Contains a list of Augmentations to be applied.
class AugmentationSequence(Augmentation):
    def __init__(self, augmentations, probability=1.0):
        super().__init__(probability=probability)
        self.augmentations = augmentations

  def __repr__(self):
    return f"LowInkLineAugmentation(use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

  def add_transparency_line(self, mask, y, alpha=None):
    if (alpha == None):
      alpha = random.randint(16, 224)

    if (self.use_consistent_lines):
      low_ink_line = np.full(mask[y, :].shape, alpha, dtype="uint8")
    else:
      low_ink_line = self.inconsistent_transparency_line(mask[y, :])

    mask[y, :] = self.apply_line(mask[y, :], low_ink_line)

    return mask

# Given a list of Augmentations, selects one to apply.
class OneOf(Augmentation):
    def __init__(self, augmentations, probability=0.5):
        super().__init__(probability=probability)
        self.augmentations = augmentations

        # Compute the average probability from all augmentations.
        augmentation_probabilities = [
            augmentation.probability for augmentation in augmentations
        ]
        s = sum(augmentation_probabilities)
        self.augmentation_probabilities = [ap / s for ap in augmentation_probabilities]

    # Randomly selects an Augmentation to apply to data.
    def __call__(self, data, force=False):
        if self.augmentation_probabilities and (force or self.should_run()):

            # Seed the random object.
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            # Randomly selects one Augmentation to apply.
            augmentation = random_state.choice(
                self.augmentations, p=self.augmentation_probabilities
            )

            # Applies the selected Augmentation.
            augmentation(data, force=True)

    # Constructs a string containing the representations
    # of each augmentation
    def __repr__(self):
        r = f"OneOf([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], probability={self.probability})"
        return r

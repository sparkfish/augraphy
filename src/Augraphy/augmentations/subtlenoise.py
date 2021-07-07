################################################################################
# File: subtlenoise.py
#
# Class: SubtleNoiseAugmentation
#
# Description: This file contains a class defining an Augmentation which
#              emulates the imperfections in scanning solid colors due to subtle
#              lighting differences.adjusts
################################################################################


################################################################################
# Imports
################################################################################

import numpy as np
import random

from Augraphy.Augmentations import *


################################################################################
# Class Definition
################################################################################


class SubtleNoiseAugmentation(Augmentation):
    def __init__(self, range=10, probability=0.5):
        super().__init__(probability=probability)
        self.range = range
        self.add_subtle_noise = np.vectorize(
            lambda x: max(0, min(255, x + random.randint(-self.range, self.range)))
        )

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SubtleNoiseAugmentation(range={self.range}, probability={self.probability})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["post"][-1].result
            data["post"].append(AugmentationResult(self, self.add_subtle_noise(image)))

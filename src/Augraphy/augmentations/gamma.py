################################################################################
# File: gamma.py
#
# Class: GammaAugmentation
#
# Description: This file contains a class defining an Augmentation which adjusts
#              the gamma of the whole image by a chosen multiplier.
################################################################################


################################################################################
# Imports
################################################################################

import cv2
import numpy as np
import random
import os

from Augraphy.Augmentations import *


################################################################################
# Class Definition
################################################################################


class GammaAugmentation(Augmentation):
    def __init__(self, range=(0.5, 1.5), probability=0.5):
        super().__init__(probability=probability)
        self.range = range

    def __repr__(self):
        return f"GammaAugmentation(range={self.range}, probability={self.probability})"

    def __call__(self, data, force=False):
        img = data["post"][-1].result
        img = img.astype(np.uint8)
        value = random.uniform(self.range[0], self.range[1])
        invGamma = 1.0 / value
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        print(table.shape, img.shape)
        frame = cv2.LUT(img, table)
        data["post"].append(AugmentationResult(self, frame))
        # cv2.imwrite(os.path.join('test_outputs',str(round(value,4))+"gamma.jpg"),np.hstack((img,frame)))

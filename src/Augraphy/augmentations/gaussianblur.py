################################################################################
# File: gaussianblur.py
#
# Class: GaussianBlurAugmentation
#
# Description: This file contains a class defining an Augmentation which applies
#              a gaussian blur to the whole image.
################################################################################


################################################################################
# Imports
################################################################################

import cv2
import random

from Augraphy.Augmentations import *


################################################################################
# Class Definition
################################################################################


class GaussianBlurAugmentation(Augmentation):
    def __init__(self, layer, kernels=[(3, 3)], sigmaX=0, probability=0.5):
        super().__init__(probability=probability)
        self.sigmaX = sigmaX
        self.kernels = kernels
        self.layer = layer

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data[self.layer][-1].result
            img = cv2.GaussianBlur(img, random.choice(self.kernels), self.sigmaX)
            data[self.layer].append(AugmentationResult(self, img))

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"GaussianBlurAugmentation({self.layer}, kernels={self.kernels}, sigmaX={self.sigmaX}, probability={self.probability})"

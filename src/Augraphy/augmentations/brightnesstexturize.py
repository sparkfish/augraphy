################################################################################
# File: brightnesstexturize.py
#
# Class: BrightnessTexturizeAugmentation
#
# Description: This file contains a class defining an Augmentation which creates
#              a random noise in the brightness channel to emulate paper
#              textures.
################################################################################


################################################################################
# Imports
################################################################################

import cv2
import numpy as np
import random

from Augraphy.Augmentations import *


################################################################################
# Class Definition
################################################################################


class BrightnessTexturizeAugmentation(Augmentation):
    def __init__(
        self, range=(0.9, 0.99), deviation=0.03, layer="paper", probability=0.5
    ):
        super().__init__(probability=probability)
        self.low = range[0]
        self.high = range[1]
        self.deviation = deviation
        self.layer = layer
        self.range = range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BrightnessTexturizeAugmentation(layer='{self.layer}', range={self.range}, deviation={self.deviation}, probability={self.probability})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data[self.layer][-1].result
            value = random.uniform(self.low, self.high)
            hsv = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)

            low = value - (value * self.deviation)  # *random.uniform(0, deviation)
            max = value + (value * self.deviation)
            brightness_matrix = np.random.uniform(
                low, max, (hsv.shape[0], hsv.shape[1])
            )
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            hsv = np.array(hsv, dtype=np.uint8)
            hsv = cv2.bitwise_not(hsv)
            hsv = np.array(hsv, dtype=np.float64)

            low = value - (value * self.deviation)
            max = value + (value * self.deviation)
            brightness_matrix = np.random.uniform(
                low, max, (hsv.shape[0], hsv.shape[1])
            )
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            hsv = np.array(hsv, dtype=np.uint8)
            hsv = cv2.bitwise_not(hsv)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            data[self.layer].append(AugmentationResult(self, img))

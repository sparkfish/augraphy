import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class Brightness(Augmentation):
    """Adjusts the brightness of the whole image by a chosen multiplier.

    :param range: Pair of ints determining the range from which to sample
           the brightness shift.
    :type range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, range=(0.8, 1.4), p=1):
        """Constructor method"""
        super().__init__(p=p)
        self.range = range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Brightness(range={self.range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            value = random.uniform(self.range[0], self.range[1])
            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2HSV)

            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 1] = hsv[:, :, 1] * value
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2] = hsv[:, :, 2] * value
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return image

import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class Brightness(Augmentation):
    """Adjusts the brightness of the whole image by a chosen multiplier.

    :param brightness_range: Pair of ints determining the range from which to sample
           the brightness shift.
    :type brightness_range: tuple, optional
    :param min_brightness: Flag to enable min brightness intensity value in
            the augmented image.
    :type min_brightness: int, optional
    :param min_brightness_value: Pair of ints determining the minimum
            brightness intensity of augmented image.
    :type min_brightness_value: tuple, optional

    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        brightness_range=(0.8, 1.4),
        min_brightness=0,
        min_brightness_value=(120, 150),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.min_brightness = min_brightness
        self.min_brightness_value = min_brightness_value

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Brightness(brightness_range={self.brightness_range}, min_brightness={self.min_brightness}, min_brightness_value={self.min_brightness_value}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            value = random.uniform(self.brightness_range[0], self.brightness_range[1])
            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2HSV)

            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * value

            # increase intensity value for area with intensity below min brightness value
            if self.min_brightness:
                min_brightness_value = min(
                    255,
                    random.randint(self.min_brightness_value[0], self.min_brightness_value[1]),
                )
                counting_step = 10
                counting_value = counting_step
                while counting_value < min_brightness_value:
                    hsv[:, :, 2][hsv[:, :, 2] < counting_value] += counting_step
                    counting_value += counting_step

            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            return image

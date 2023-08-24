import random

import cv2
import numpy as np
from numba import config
from numba import jit

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
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        brightness_range=(0.8, 1.4),
        min_brightness=0,
        min_brightness_value=(20, 50),
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p, numba_jit=numba_jit)
        self.brightness_range = brightness_range
        self.min_brightness = min_brightness
        self.min_brightness_value = min_brightness_value
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Brightness(brightness_range={self.brightness_range}, min_brightness={self.min_brightness}, min_brightness_value={self.min_brightness_value}, numba_jit={self.numba_jit}, p={self.p})"

    @staticmethod
    @jit(nopython=True, cache=True)
    def adjust_min_brightness(image, min_brightness_value):
        """Increase image pixel intensity by value of 10 in each iteration until reaching the min brightness value.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param min_brightness_value: The minimum brightness of value of each pixel.
        :type min_brightness_value: int

        """

        ysize, xsize = image.shape[:2]
        image_flat = image.ravel()

        counting_step = 10.0
        counting_value = counting_step
        while counting_value < min_brightness_value:
            indices = image_flat < counting_value
            image_flat[indices] += counting_step
            counting_value += counting_step

        indices = image_flat > 255
        image_flat[indices] = 255
        image = image_flat.reshape(ysize, xsize)

        return image

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            value = random.uniform(self.brightness_range[0], self.brightness_range[1])

            hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2HSV)

            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * value
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # increase intensity value for area with intensity below min brightness value
            if self.min_brightness:
                min_brightness_value = min(
                    255,
                    random.randint(self.min_brightness_value[0], self.min_brightness_value[1]),
                )

                v = self.adjust_min_brightness(hsv[:, :, 2], min_brightness_value)
                hsv[:, :, 2] = v

            hsv = np.array(hsv, dtype=np.uint8)
            image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            return image_output

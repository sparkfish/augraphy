import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class BrightnessTexturize(Augmentation):
    """Creates a random noise in the brightness channel to emulate paper
    textures.

    :param texturize_range: Pair of floats determining the range from which to sample values
           for the brightness matrix. Suggested value = <1.
    :type brightness_range: tuple, optional
    :param deviation: Additional variation for the uniform sample.
    :type deviation: float, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, texturize_range=(0.8, 0.99), deviation=0.08, p=1):
        """Constructor method"""
        super().__init__(p=p)
        self.low = texturize_range[0]
        self.high = texturize_range[1]
        self.deviation = deviation
        self.texturize_range = texturize_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BrightnessTexturize(texturize_range={self.texturize_range}, deviation={self.deviation}, p={self.p})"

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

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # compute random value
            value = random.uniform(self.low, self.high)
            # convert to float
            hsv = np.array(hsv, dtype=np.float64)

            # add noise using deviation
            low_value = value - (value * self.deviation)  # *random.uniform(0, deviation)
            max_value = value + (value * self.deviation)

            # apply noise
            brightness_matrix = np.random.uniform(low_value, max_value, size=(hsv.shape[0], hsv.shape[1]))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # convert back to uint8, apply bitwise not and convert to hsv again
            hsv = np.array(hsv, dtype=np.uint8)

            # non hue and saturation channel to prevent color change
            hsv[:, :, 2] = cv2.bitwise_not(hsv[:, :, 2])
            hsv = hsv.astype("float64")

            # add noise using deviation again
            low_value = value - (value * self.deviation)
            max_value = value + (value * self.deviation)

            # apply noise again
            brightness_matrix = np.random.uniform(low_value, max_value, size=(hsv.shape[0], hsv.shape[1]))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # convert back to uint8, apply bitwise not
            hsv = np.array(hsv, dtype=np.uint8)
            # non hue and saturation channel to prevent color change
            hsv[:, :, 2] = cv2.bitwise_not(hsv[:, :, 2])

            image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            return image_output

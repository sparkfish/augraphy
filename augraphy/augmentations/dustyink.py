import random

import cv2
import numpy as np

from augraphy.augmentations.lib import add_noise
from augraphy.base.augmentation import Augmentation


class DustyInk(Augmentation):
    """Applies random noise to the ink itself, emulating a dusty or
    inconsistent ink tone when followed by a blur.

    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    :param value_range: Min value of pixel to enable dusty ink effect.
    :type value_range: tuple, optional
    :param p: Probability of this Augmentation being applied.
    :type p: float, optional
    """

    def __init__(
        self,
        intensity_range=(0.1, 0.2),
        color_range=(0, 224),
        value_range=(0, 5),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = list(intensity_range)
        self.color_range = list(color_range)
        self.value_range = list(value_range)

        # prevent second range value > first range value
        self.intensity_range[0] = min(self.intensity_range[0], self.intensity_range[1])
        self.color_range[0] = min(self.color_range[0], self.color_range[1])
        self.value_range[0] = min(self.value_range[0], self.value_range[1])

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DustyInk(intensity_range={self.intensity_range}, color_range={self.color_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            min_value = random.randint(self.value_range[0], self.value_range[1])
            apply_mask_fn = lambda x, y: y if (x < min_value) else x
            apply_mask = np.vectorize(apply_mask_fn)
            noise_mask = add_noise(image, self.intensity_range, self.color_range)
            noise_mask = cv2.GaussianBlur(noise_mask, (3, 3), 0)
            image = apply_mask(image, noise_mask)

            return image

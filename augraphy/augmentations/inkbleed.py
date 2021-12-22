import random
import sys

import cv2
import numpy as np

from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation


class InkBleed(Augmentation):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the range from which
           noise intensity is sampled.
    :type intensity: tuple, optional
    :param color_range: Pair of ints determining the range from which color
           noise is sampled.
    :type color_range: tuple, optional
    :param kernel_size: Kernel size to determine area of inkbleed effect.
    :type kernel_size: tuple, optional
    :param severity: Severity to determine concentration of inkbleed effect.
    :type severity: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        intensity_range=(0.1, 0.2),
        color_range=(0, 224),
        kernel_size=(5, 5),
        severity=(0.4, 0.6),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_range = color_range
        self.kernel_size = kernel_size
        self.severity = severity

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"InkBleed(intensity_range={self.intensity_range}, color_range={self.color_range}, kernel_size={self.kernel_size}, severity={self.severity}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            add_noise_fn = (
                lambda x, y: random.randint(self.color_range[0], self.color_range[1])
                if (y == 255 and random.random() < intensity)
                else x
            )

            severity = random.randint(self.severity[0] * 10, self.severity[1] * 10) / 10
            apply_mask_fn = lambda x, y, z: x if (z != 255 or random.random() > severity) else y
            add_noise = np.vectorize(add_noise_fn)
            apply_mask = np.vectorize(apply_mask_fn)
            sobelized = sobel(image)

            kernel = np.ones(self.kernel_size, dtype="uint8")
            sobelizedDilated = cv2.dilate(sobelized, kernel, iterations=1)

            noise_mask = add_noise(image, sobelizedDilated)
            noise_mask = noise_mask.astype("uint8")
            noise_mask = cv2.GaussianBlur(noise_mask, (3, 3), 0)

            image = apply_mask(image, noise_mask, sobelizedDilated)

            return image

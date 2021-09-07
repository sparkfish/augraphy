import random

import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Downscale(Augmentation):
    """Uses Downscale the image to introduce resizing artifacts in the image.

    :param scale_range: Pair of ints determining the range from which to
           divide the resolution by.
    :type scale_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, scale_range=(1.1, 3), p=0.5):
        """Constructor method"""
        super().__init__(p=p)
        self.scale_range = scale_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Downscale(scale_range={self.scale_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["post"][-1].result
            scale = random.randint(self.scale_range[0], self.scale_range[1])
            new_size = (image.shape[1] // scale, image.shape[0] // scale)
            downscaled = cv2.resize(image, new_size)
            upscaled = cv2.resize(downscaled, (image.shape[1], image.shape[0]))
            data["post"].append(AugmentationResult(self, upscaled))

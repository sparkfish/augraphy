import numpy as np
import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.augmentations.lib import addNoise


class DustyInkAugmentation(Augmentation):
    """Applies random noise to the ink itself, emulating a dusty or
    inconsistent ink tone when followed by a blur.

    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    :param p: Probability of this Augmentation being applied.
    :type p: float, optional
    """

    def __init__(self, intensity_range=(0.1, 0.2), color_range=(0, 224), p=0.5):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_range = color_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DustyInkAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data["ink"][-1].result
            img = addNoise(img)
            data["ink"].append(AugmentationResult(self, img))

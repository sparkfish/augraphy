import numpy as np
import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.augmentations.lib import addNoise

class DustyInk(Augmentation):
    """Applies random noise to the ink itself, emulating a dusty or
    inconsistent ink tone when followed by a blur.

    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    :param p: Probability of this Augmentation being applied.
    :type p: float, optional
    """

    def __init__(
        self, intensity_range=(0.1, 0.2), color_range=(0, 224), p=0.5
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_range = color_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DustyInk(intensity_range={self.intensity_range}, color_range={self.color_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data["ink"][-1].result
            apply_mask_fn = (
                lambda x, y: y if (x < 64) else x
            )
            apply_mask = np.vectorize(apply_mask_fn)
            noise_mask = addNoise(img, self.intensity_range, self.color_range)
            noise_mask = cv2.GaussianBlur(noise_mask, (3,3), 0)
            img = apply_mask(img, noise_mask)
            
            data["ink"].append(AugmentationResult(self, img))

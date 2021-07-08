import numpy as np
import random

from base.augmentation import Augmentation
from base.augmentationResult import AugmentationResult


class DustyInkAugmentation(Augmentation):
    """Applies random noise to the ink itself, emulating a dusty or
    inconsistent ink tone when followed by a blur.

    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    :param probability: Probability of this Augmentation being applied.
    :type probability: float, optional
    """

    def __init__(
        self, intensity_range=(0.1, 0.2), color_range=(0, 224), probability=0.5
    ):
        """Constructor method"""
        super().__init__(probability=probability)
        self.intensity_range = intensity_range
        self.color_range = color_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DustyInkAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, probability={self.probability})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data["ink"][-1].result
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            add_noise_fn = (
                lambda x: random.randint(self.color_range[0], self.color_range[1])
                if (x == 0 and random.random() < intensity)
                else x
            )
            add_noise = np.vectorize(add_noise_fn)
            img = add_noise(img)

            data["ink"].append(AugmentationResult(self, img))

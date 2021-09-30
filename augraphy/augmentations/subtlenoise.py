import random

import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class SubtleNoise(Augmentation):
    """Emulates the imperfections in scanning solid colors due to subtle
    lighting differences.

    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param range: The possible range of noise variation to sample from.
    :type range: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        range=10,
        p=0.5,
    ):
        super().__init__(p=p)
        self.layer = layer
        self.range = range
        self.add_subtle_noise = np.vectorize(
            lambda x: max(0, min(255, x + random.randint(-self.range, self.range))),
        )

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SubtleNoise(layer={self.layer}, range={self.range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result.copy()
            data[self.layer].append(AugmentationResult(self, self.add_subtle_noise(image)))

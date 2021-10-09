import random

import numpy as np

from augraphy.base.augmentation import Augmentation


class SubtleNoise(Augmentation):
    """Emulates the imperfections in scanning solid colors due to subtle
    lighting differences.

    :param range: The possible range of noise variation to sample from.
    :type range: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        range=10,
        p=1,
    ):
        super().__init__(p=p)
        self.range = range
        self.add_subtle_noise = np.vectorize(
            lambda x: max(0, min(255, x + random.randint(-self.range, self.range))),
        )

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SubtleNoise(range={self.range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            image = self.add_subtle_noise(image)
            return image

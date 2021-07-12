import numpy as np
import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class SubtleNoiseAugmentation(Augmentation):
    """Emulates the imperfections in scanning solid colors due to subtle
    lighting differences.

    :param range: The possible range of noise variation to sample from.
    :type range: int, optional
    :param probability: The probability that this Augmentation will be applied.
    :type probability: float, optional
    """

    def __init__(self, range=10, probability=0.5):
        super().__init__(probability=probability)
        self.range = range
        self.add_subtle_noise = np.vectorize(
            lambda x: max(0, min(255, x + random.randint(-self.range, self.range)))
        )

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SubtleNoiseAugmentation(range={self.range}, probability={self.probability})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["post"][-1].result
            data["post"].append(AugmentationResult(self, self.add_subtle_noise(image)))

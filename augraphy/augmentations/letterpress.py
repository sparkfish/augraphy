import random

import cv2
import numpy as np

from augraphy.augmentations.lib import addNoise
from augraphy.augmentations.lib import applyBlob
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Letterpress(Augmentation):
    """Produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

    :param count_range: Pair of ints determining the range from which the number
           of blobs to generate is sampled.
    :type count_range: tuple, optional
    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        count_range=(1000, 2500),
        size_range=(60, 80),
        points_range=(200, 250),
        std_range=(10, 75),
        features_range=(15, 25),
        value_range=(200, 250),
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.count_range = count_range
        self.size_range = size_range
        self.points_range = points_range
        self.std_range = std_range
        self.features_range = features_range
        self.value_range = value_range

    def __repr__(self):
        return f"Letterpress(count_range={self.count_range}, size_range={self.size_range}, points_range={self.points_range}, std_range={self.std_range}, features_range={self.features_range}, value_range={self.value_range}, p={self.p})"

    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["ink"][-1].result.copy()
            count = random.randint(self.count_range[0], self.count_range[1])
            noise_mask = np.copy(image)

            for i in range(count):
                noise_mask = applyBlob(
                    noise_mask,
                    self.size_range,
                    self.points_range,
                    self.std_range,
                    self.features_range,
                    self.value_range,
                )

            apply_mask_fn = lambda x, y: y if (x < 128) else x
            apply_mask = np.vectorize(apply_mask_fn)

            noise_mask = addNoise(noise_mask)
            noise_mask = cv2.GaussianBlur(noise_mask, (3, 3), 0)
            image = apply_mask(image, noise_mask)

            data["ink"].append(AugmentationResult(self, image))

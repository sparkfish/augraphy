import os
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Gamma(Augmentation):
    """Adjusts the gamma of the whole image by a chosen multiplier.

    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param range: Pair of ints determining the range from which to sample the
           gamma shift.
    :type range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        range=(0.5, 1.5),
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.layerlayer
        self.range = range

    def __repr__(self):
        return f"Gamma(layer={self.layer}, range={self.range}, p={self.p})"

    def __call__(self, data, force=False):
        img = data[self.layer][-1].result.copy()
        img = img.astype(np.uint8)
        value = random.uniform(self.range[0], self.range[1])
        invGamma = 1.0 / value
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)],
        ).astype("uint8")
        print(table.shape, img.shape)
        frame = cv2.LUT(img, table)
        data[self.layer].append(AugmentationResult(self, frame))
        # cv2.imwrite(os.path.join('test_outputs',str(round(value,4))+"gamma.jpg"),np.hstack((img,frame)))

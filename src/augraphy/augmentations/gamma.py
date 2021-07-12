import cv2
import numpy as np
import random
import os

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult

class GammaAugmentation(Augmentation):
    """Adjusts the gamma of the whole image by a chosen multiplier.

    :param range: Pair of ints determining the range from which to sample the
           gamma shift.
    :type range: tuple, optional
    :param probability: The probability that this Augmentation will be applied.
    :type probability: float, optional
    """

    def __init__(self, range=(0.5, 1.5), probability=0.5):
        """Constructor method"""
        super().__init__(probability=probability)
        self.range = range

    def __repr__(self):
        return f"GammaAugmentation(range={self.range}, probability={self.probability})"

    def __call__(self, data, force=False):
        img = data["post"][-1].result
        img = img.astype(np.uint8)
        value = random.uniform(self.range[0], self.range[1])
        invGamma = 1.0 / value
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        print(table.shape, img.shape)
        frame = cv2.LUT(img, table)
        data["post"].append(AugmentationResult(self, frame))
        # cv2.imwrite(os.path.join('test_outputs',str(round(value,4))+"gamma.jpg"),np.hstack((img,frame)))

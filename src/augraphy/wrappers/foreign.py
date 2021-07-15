"""This file contains wrapper classes for augmentations from other projects.

Currently supported:
---
1. Albumentations: https://github.com/albumentations-team/albumentations
2. Imgaug: https://github.com/aleju/imgaug
"""

import numpy


from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class ForeignAugmentation(Augmentation):
    """A wrapper for augmentations from other projects.

    :param layer: The Augraphy layer this augmentation should be applied in.
    :type layer: string
    :param foreignAugmentation: The fully-applied constructor for the foreign transform.
    :type foreignAugmentation: object
    :param p: The probability that augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, layer, foreignAugmentation, p=0.5):
        self.layer = layer
        self.augmentation = foreignAugmentation
        super().__init__(p=p)

    def __call__(self, data):
        image = data[self.layer][-1].result
        result = self.augmentation(image = image)
        output = self.handleForeignAugResult(result)
        data[self.layer].append(AugmentationResult(self, output))

    def handleForeignAugResult(*res):
        """The argument to this depends on the foreign augmentation applied.

        If an Albumentation augmentation is used, the result is a dict with the
        output image at key "image".

        If an imgaug augmenter is used, the result is a numpy.ndarray.
        """
        if isinstance(res[1], dict):
            return res[1]["image"]
        else:
            return res[1]

"""This file contains a wrapper class for augmentations from other projects.
Currently supported:
---
1. Albumentations: https://github.com/albumentations-team/albumentations
2. Imgaug: https://github.com/aleju/imgaug
"""
import numpy

from augraphy.base.augmentation import Augmentation


class ForeignAugmentation(Augmentation):
    """A wrapper for augmentations from other projects.

    :param foreignAugmentation: The fully-applied constructor for the foreign transform.
    :type foreignAugmentation: object
    :param p: The probability that augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, foreignAugmentation, p=1):
        self.augmentation = foreignAugmentation
        super().__init__(p=p)

    def __call__(self, image):
        image = image.copy()
        result = self.augmentation(image=image)
        output = self.handleForeignAugResult(result)
        return output

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

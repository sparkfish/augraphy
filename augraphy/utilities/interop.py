"""This file contains a wrapper class for augmentations from other projects.
Currently supported:
---
1. Albumentations: https://github.com/albumentations-team/albumentations
2. Imgaug: https://github.com/aleju/imgaug
"""
import numpy

from augraphy.base.augmentation import Augmentation


class Interop(Augmentation):
    """A wrapper for augmentations from other projects.

    :param augmentation: The fully-applied constructor for the interop transform.
    :type augmentation: object
    :param p: The probability that augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, augmentation, p=1):
        self.augmentation = augmentation
        super().__init__(p=p)

    def __call__(self, image, layer=None, force=False):

        image = image.copy()
        result = self.augmentation(image=image)
        output = self.handle_augmentation_result(result)
        return output

    def handle_augmentation_result(*res):
        """The argument to this depends on the augmentation applied.
        If an Albumentation augmentation is used, the result is a dict with the
        output image at key "image".
        If an imgaug augmenter is used, the result is a numpy.ndarray.
        """
        if isinstance(res[1], dict):
            return res[1]["image"]
        else:
            return res[1]

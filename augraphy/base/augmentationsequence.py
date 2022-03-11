from augraphy.base.augmentation import Augmentation


class AugmentationSequence(Augmentation):
    """A list of Augmentations to be applied sequentially.

    :param augmentations: A list of Augmentation objects to be applied.
    :type augmentations: list
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, augmentations, p=1):
        """Constructor method"""
        self.augmentations = augmentations
        self.p = p

    def __repr__(self):
        output = "AugmentationSequence([\n"
        for aug in self.augmentations:
            output += f"\t{repr(aug)},\n"

        output += "])"
        return output

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image
            for augmentation in self.augmentations:
                result = augmentation(result)

            return result, self.augmentations

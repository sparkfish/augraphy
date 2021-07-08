class AugmentationSequence:
    """A list of Augmentations to be applied sequentially.

    :param augmentations: A list of Augmentation objects to be applied.
    :type augmentations: list
    """
    def __init__(self, augmentations):
        """Constructor method"""
        self.augmentations = augmentations

    def __repr__(self):
        output = f"AugmentationSequence([\n"
        for aug in augmentations:
            output += f"\t{repr(aug)}\n"

        output += "])"
        return output

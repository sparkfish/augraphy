import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationsequence import AugmentationSequence


class OneOf(Augmentation):
    """Given a list of Augmentations, selects one to apply.

    :param augmentations: A list of Augmentations to choose from.
    :type augmentations: list
    :param p: The probability that this augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, augmentations, p=1):
        """Constructor method"""
        self.augmentations = augmentations
        self.augmentation_probabilities = self.computeProbability(self.augmentations)
        self.p = p

    # Randomly selects an Augmentation to apply to data.
    def __call__(self, image, layer=None, force=False):
        if self.augmentation_probabilities and (force or self.should_run()):

            # Randomly selects one Augmentation to apply.
            augmentation = random.choice(self.augmentations)

            # Applies the selected Augmentation.
            image = augmentation(image, force=True)
            return image

    # Constructs a string containing the representations
    # of each augmentation
    def __repr__(self):
        r = "OneOf([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], p={self.p})"
        return r

    def computeProbability(self, augmentations):
        """For each Augmentation in the input list, compute the probability of
        applying that Augmentation.

        :param augmentations: Augmentations to compute probability list for.
        :type augmentations: list
        """

        augmentation_probabilities = [augmentation.p for augmentation in augmentations]
        s = sum(augmentation_probabilities)
        return [ap / s for ap in augmentation_probabilities]

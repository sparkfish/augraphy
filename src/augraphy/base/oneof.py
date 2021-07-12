import numpy as np
import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationsequence import AugmentationSequence
from augraphy.base.augmentation import Augmentation

class OneOf(Augmentation):
    """Given a list of Augmentations, selects one to apply.

    :param augmentations: A list of Augmentations to choose from.
    :type augmentations: list
    """

    def __init__(self, augmentations, probability=0.5):
        """Constructor method"""
        self.augmentations = augmentations
        self.augmentation_probabilities = self.computeProbability(self.augmentations)
        self.probability = probability

    # Randomly selects an Augmentation to apply to data.
    def __call__(self, data, force=False):
        if self.augmentation_probabilities and (force or self.should_run()):

            # Seed the random object.
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            # Randomly selects one Augmentation to apply.
            augmentation = random_state.choice(
                self.augmentations, p=self.augmentation_probabilities
            )

            # Applies the selected Augmentation.
            augmentation(data, force=True)

    # Constructs a string containing the representations
    # of each augmentation
    def __repr__(self):
        r = f"OneOf([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], probability={self.probability})"
        return r

    def computeProbability(self, augmentations):
        """For each Augmentation in the input list, compute the probability of
        applying that Augmentation.

        :param augmentations: Augmentations to compute probability list for.
        :type augmentations: list
        """
        augmentation_probabilities = []

        """Allow for both AugmentationSequences and Augmentations"""
        for aug in augmentations:
            if isinstance(aug, AugmentationSequence):
                for a in aug.augmentations:
                    augmentation_probabilities.append(a.probability)

            if isinstance(aug, Augmentation):
                augmentation_probabilities.append(aug.probability)

        s = sum(augmentation_probabilities)
        return [ap / s for ap in augmentation_probabilities]

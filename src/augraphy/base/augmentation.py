import random


class Augmentation:
    """The base class which all pipeline augmentations inherit from.

    :param probability: The probability that this augmentation will be run when
        executed as part of a pipeline.
    :type probability: float, optional
    """

    def __init__(self, probability=0.5):
        """Constructor method"""
        self.probability = probability

    def should_run(self):
        """Determines whether or not the augmentation should be applied
        by callers.

        :return: True if the probability given was no smaller than the
            random sample on the unit interval.
        :rtype: bool
        """
        return random.uniform(0.0, 1.0) <= self.probability

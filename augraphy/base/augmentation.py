import random


class Augmentation:
    """The base class which all pipeline augmentations inherit from.

    :param p: The probability that this augmentation will be run when executed as part of a pipeline.
    :type p: float, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    """

    def __init__(self, p=0.5, numba_jit=1):
        """Constructor method"""
        self.p = p
        self.numba_jit = numba_jit

    def should_run(self):
        """Determines whether or not the augmentation should be applied
        by callers.

        :return: True if the probability given was no smaller than the
            random sample on the unit interval.
        :rtype: bool
        """
        return random.uniform(0.0, 1.0) <= self.p

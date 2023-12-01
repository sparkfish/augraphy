import random


class Augmentation:
    """The base class which all pipeline augmentations inherit from.

    :param mask: The mask of labels for each pixel. Mask value should be in range of 0 to 255.
    :type mask: numpy array (uint8), optional
    :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate (x, y).
    :type keypoints: dictionary, optional
    :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
    :type bounding_boxes: list, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this augmentation will be run when executed as part of a pipeline.
    :type p: float, optional
    """

    def __init__(self, mask=None, keypoints={}, bounding_boxes=[], p=0.5, numba_jit=1):
        """Constructor method"""

        self.mask = mask
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.numba_jit = numba_jit
        self.p = p

    def should_run(self):
        """Determines whether or not the augmentation should be applied
        by callers.

        :return: True if the probability given was no smaller than the
            random sample on the unit interval.
        :rtype: bool
        """
        return random.uniform(0.0, 1.0) <= self.p

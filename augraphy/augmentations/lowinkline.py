################################################################################
# File: lowinkline.py
#
import random

import numpy as np

from augraphy.base.augmentation import Augmentation


class LowInkLine(Augmentation):
    """Generates streaking behavior common to printers running out of ink.

    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, use_consistent_lines=True, p=0.5):
        """Constructor method"""
        super().__init__(p=p)

        self.use_consistent_lines = use_consistent_lines
        inconsistent_transparency_line = lambda x: random.randint(0, 255)
        self.inconsistent_transparency_line = np.vectorize(
            inconsistent_transparency_line,
        )

        apply_line = lambda x, y: x if x > y else y
        self.apply_line = np.vectorize(apply_line)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return (
            f"LowInkLine(use_consistent_lines={self.use_consistent_lines}, p={self.p})"
        )

    # Takes an image, a vertical position, and an opacity value,
    # then adds a line at that position in the image with the given
    # opacity.
    def add_transparency_line(self, mask, y, alpha=None):
        """Adds a line with some opacity at a vertical position in the image.

        :param mask: The image to apply the line to.
        :type mask: numpy.array
        :param y: The vertical position to apply the line at.
        :type y: int
        :param alpha: The desired opacity of the line.
        :type alpha: int, optional
        """
        if alpha is None:
            alpha = random.randint(16, 224)

        if self.use_consistent_lines:
            low_ink_line = np.full(mask[y, :].shape, alpha, dtype="uint8")
        else:
            low_ink_line = self.inconsistent_transparency_line(mask[y, :])

        mask[y, :] = self.apply_line(mask[y, :], low_ink_line)

        return mask

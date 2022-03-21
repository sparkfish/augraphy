################################################################################
# File: lowinkline.py
#
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class LowInkLine(Augmentation):
    """Generates streaking behavior common to printers running out of ink.

    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        use_consistent_lines=True,
        noise_probability=0.1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.use_consistent_lines = use_consistent_lines
        self.noise_probability = noise_probability

        inconsistent_transparency_line = lambda x: random.randint(0, 255)
        self.inconsistent_transparency_line = np.vectorize(
            inconsistent_transparency_line,
        )

        inconsistent_transparency_line_topbottom = (
            lambda x: random.randint(0, 255) if random.random() > (1 - self.noise_probability) else x
        )
        self.inconsistent_transparency_line_topbottom = np.vectorize(
            inconsistent_transparency_line_topbottom,
        )

        consistent_transparency_line_topbottom = lambda x, y: y if random.random() > (1 - self.noise_probability) else x
        self.consistent_transparency_line_topbottom = np.vectorize(
            consistent_transparency_line_topbottom,
        )

        apply_line = lambda x, y: x if x > y else y
        self.apply_line = np.vectorize(apply_line)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkLine(use_consistent_lines={self.use_consistent_lines}, p={self.p})"

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

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                low_ink_line_top = self.consistent_transparency_line_topbottom(
                    mask[y - 1, :],
                    alpha,
                )
                if len(mask.shape) > 2:
                    low_ink_line_top = np.dstack(
                        [
                            low_ink_line_top[:, 0],
                            low_ink_line_top[:, 0],
                            low_ink_line_top[:, 0],
                        ],
                    )[0]

            if y + 1 < mask.shape[0]:
                low_ink_line_bottom = self.consistent_transparency_line_topbottom(
                    mask[y + 1, :],
                    alpha,
                )
                if len(mask.shape) > 2:
                    low_ink_line_bottom = np.dstack(
                        [
                            low_ink_line_bottom[:, 0],
                            low_ink_line_bottom[:, 0],
                            low_ink_line_bottom[:, 0],
                        ],
                    )[0]

        else:
            low_ink_line = self.inconsistent_transparency_line(mask[y, :])

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                low_ink_line_top = self.inconsistent_transparency_line_topbottom(
                    mask[y - 1, :],
                )
            if y + 1 < mask.shape[0]:
                low_ink_line_bottom = self.inconsistent_transparency_line_topbottom(
                    mask[y + 1, :],
                )

        mask[y, :] = self.apply_line(mask[y, :], low_ink_line)
        # apply noise to top and bottom of the line
        if y - 1 >= 0:
            mask[y - 1, :] = self.apply_line(mask[y - 1, :], low_ink_line_top)
        if y + 1 < mask.shape[0]:
            mask[y + 1, :] = self.apply_line(mask[y + 1, :], low_ink_line_bottom)

        return mask

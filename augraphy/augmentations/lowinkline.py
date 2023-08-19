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
        ysize, xsize = mask.shape[:2]

        if alpha is None:
            alpha = random.randint(16, 224)

        if self.use_consistent_lines:
            low_ink_line = np.full(mask[y, :].shape, alpha, dtype="uint8")

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                if len(mask.shape) > 2:
                    indices = np.random.random((xsize, mask.shape[2])) > (1 - self.noise_probability)
                else:
                    indices = np.random.random((xsize)) > (1 - self.noise_probability)
                low_ink_line_top = mask[y - 1, :].copy()
                low_ink_line_top[indices] = alpha

                if len(mask.shape) > 2:
                    stacked_line = [
                        low_ink_line_top[:, 0],
                        low_ink_line_top[:, 0],
                        low_ink_line_top[:, 0],
                    ]
                    if mask.shape[2] == 4:
                        stacked_line += [low_ink_line_top[:, 3]]
                    low_ink_line_top = np.dstack(stacked_line)[0]

            if y + 1 < mask.shape[0]:
                if len(mask.shape) > 2:
                    indices = np.random.random((xsize, mask.shape[2])) > (1 - self.noise_probability)
                else:
                    indices = np.random.random((xsize)) > (1 - self.noise_probability)
                low_ink_line_bottom = mask[y + 1, :].copy()
                low_ink_line_bottom[indices] = alpha

                if len(mask.shape) > 2:
                    stacked_line = [
                        low_ink_line_bottom[:, 0],
                        low_ink_line_bottom[:, 0],
                        low_ink_line_bottom[:, 0],
                    ]
                    if mask.shape[2] == 4:
                        stacked_line += [low_ink_line_bottom[:, 3]]
                    low_ink_line_bottom = np.dstack(stacked_line)[0]

        else:
            low_ink_line = (np.random.random((xsize)) * 255).astype("uint8")
            if len(mask.shape) > 2:
                new_low_ink_line = np.zeros((1, xsize, mask.shape[2]), dtype="uint8")
                for i in range(mask.shape[2]):
                    new_low_ink_line[:, :, i] = low_ink_line.copy()
                low_ink_line = new_low_ink_line

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                indices = np.random.random((xsize)) <= (1 - self.noise_probability)
                low_ink_line_top = (np.random.random((xsize)) * 255).astype("uint8")
                if len(mask.shape) > 2:
                    new_low_ink_line_top = np.zeros((1, xsize, mask.shape[2]), dtype="uint8")
                    for i in range(mask.shape[2]):
                        new_low_ink_line_top[:, :, i] = low_ink_line_top.copy()
                        new_low_ink_line_top[:, :, i][indices.reshape(1, xsize)] = mask[y - 1, :, i][indices]
                    low_ink_line_top = new_low_ink_line_top
                else:
                    low_ink_line_top[indices] = mask[y - 1, :][indices]

            if y + 1 < mask.shape[0]:
                indices = np.random.random((xsize)) <= (1 - self.noise_probability)
                low_ink_line_bottom = (np.random.random((xsize)) * 255).astype("uint8")
                if len(mask.shape) > 2:
                    new_low_ink_line_bottom = np.zeros((1, xsize, mask.shape[2]), dtype="uint8")
                    for i in range(mask.shape[2]):
                        new_low_ink_line_bottom[:, :, i] = low_ink_line_bottom.copy()
                        new_low_ink_line_bottom[:, :, i][indices.reshape(1, xsize)] = mask[y - 1, :, i][indices]
                    low_ink_line_bottom = new_low_ink_line_bottom
                else:
                    low_ink_line_bottom[indices] = mask[y - 1, :][indices]

        indices = mask[y, :] < low_ink_line
        if len(mask.shape) > 2:
            mask[y, :, :3][indices[:, :3]] = low_ink_line[:, :3][indices[:, :3]]
        else:
            mask[y, :][indices] = low_ink_line[indices]

        if y - 1 >= 0:
            indices = mask[y - 1, :] < low_ink_line_top
            if len(mask.shape) > 2:
                mask[y - 1, :, :3][indices[:, :3]] = low_ink_line_top[:, :3][indices[:, :3]]
            else:
                mask[y - 1, :][indices] = low_ink_line_top[indices]

        if y + 1 < mask.shape[0]:
            indices = mask[y - 1, :] < low_ink_line_bottom
            if len(mask.shape) > 2:
                mask[y + 1, :, :3][indices[:, :3]] = low_ink_line_bottom[:, :3][indices[:, :3]]
            else:
                mask[y + 1, :][indices] = low_ink_line_bottom[indices]

        return mask

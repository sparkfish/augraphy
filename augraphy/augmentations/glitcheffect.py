import random

import cv2
import numpy as np

from augraphy.augmentations.colorshift import ColorShift
from augraphy.base.augmentation import Augmentation


class GlitchEffect(Augmentation):
    """Create glitch effect by applying ColorShift and shifts patches of image horizontally or vertically.

    :param glitch_direction: Direction of the glitch effect, select from "vertical", "horizontal", "all" or "random".
    :type glitch_direction: string, optional
    :param glitch_number_range: Tuple of ints determing the number of shifted image patches.
    :type glitch_number_range: tuple, optional
    :param glitch_size_range: Tuple of ints/floats determing the size of image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image height:
            size (int) = image height  * size (float and 0.0 - 1.0)
    :type glitch_size_range: tuple, optional
    :param glitch_offset_range: Tuple of ints/floats determing the offset value to shift the image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image width:
            offset (int) = image width  * offset (float and 0.0 - 1.0)
    :type glitch_offset_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        glitch_direction="random",
        glitch_number_range=(8, 16),
        glitch_size_range=(5, 50),
        glitch_offset_range=(10, 50),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.glitch_direction = glitch_direction
        self.glitch_number_range = glitch_number_range
        self.glitch_size_range = glitch_size_range
        self.glitch_offset_range = glitch_offset_range

    def __repr__(self):
        return f"GlitchEffect(glitch_direction={self.glitch_direction}, glitch_number_range={self.glitch_number_range}, glitch_size_range={self.glitch_size_range}, glitch_offset_range={self.glitch_offset_range}, p={self.p})"

    def apply_glitch(self, image):
        """Apply glitch effect into the image by shifting patches of images.

        :param image: Image to apply the glitch effect.
        :type image: numpy array
        """

        ysize, xsize = image.shape[:2]
        glitch_number = random.randint(self.glitch_number_range[0], self.glitch_number_range[1])
        for i in range(glitch_number):

            # generate random glitch size
            if self.glitch_size_range[0] <= 1.0 and isinstance(self.glitch_size_range[0], float):
                glitch_size = random.randint(
                    int(self.glitch_size_range[0] * ysize),
                    int(self.glitch_size_range[1] * ysize),
                )
            else:
                glitch_size = random.randint(self.glitch_size_range[0], self.glitch_size_range[1])

            # generate random direction
            direction = random.choice([-1, 1])

            # generate random glitch offset
            if self.glitch_offset_range[0] <= 1.0 and isinstance(self.glitch_offset_range[0], float):
                glitch_offset = (
                    random.randint(int(self.glitch_offset_range[0] * xsize), int(self.glitch_offset_range[1] * xsize))
                    * direction
                )
            else:
                glitch_offset = random.randint(self.glitch_offset_range[0], self.glitch_offset_range[1]) * direction

            # get a patch of image
            start_y = random.randint(0, ysize - glitch_size)
            image_patch = image[start_y : start_y + glitch_size, :]
            pysize, pxsize = image_patch.shape[:2]

            # create translation matrix in horizontal direction
            translation_matrix = np.float32([[1, 0, glitch_offset], [0, 1, 0]])

            # get a copy of translated area
            if direction > 0:
                image_patch_fill = image_patch[:, -glitch_offset:].copy()
            else:
                image_patch_fill = image_patch[:, :glitch_offset].copy()
            # translate image
            image_patch = cv2.warpAffine(image_patch, translation_matrix, (pxsize, pysize))
            # fill back the empty are after translation
            if direction > 0:
                image_patch[:, :glitch_offset] = image_patch_fill
            else:
                image_patch[:, -glitch_offset:] = image_patch_fill

            # randomly scale single channel to create a single color contrast effect
            random_ratio = random.uniform(0.8, 1.2)
            channel = random.randint(0, 2)
            image_patch_ratio = image_patch[:, :, channel].astype("int") * random_ratio
            image_patch_ratio[image_patch_ratio > 255] = 255
            image_patch_ratio[image_patch_ratio < 0] = 0
            image_patch[:, :, channel] = image_patch_ratio.astype("uint8")

            image[start_y : start_y + glitch_size, :] = image_patch

        return image

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # check and convert image into BGR format
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

            # apply color shift before the glitch effect
            color_shift = ColorShift(
                color_shift_offset_x_range=(3, 5),
                color_shift_offset_y_range=(3, 5),
                color_shift_iterations=(1, 2),
                color_shift_brightness_range=(0.9, 1.1),
                color_shift_gaussian_kernel_range=(1, 3),
                p=1,
            )
            image_output = color_shift(image)

            # check and generate random direction
            if self.glitch_direction == "random":
                glitch_direction = random.choice(["vertical", "horizontal"])
            else:
                glitch_direction = self.glitch_direction

            # for vertical direction, rotate image by 90 degree because apply_glitch create glithces horizontally
            if glitch_direction == "vertical":
                image_output = np.rot90(self.apply_glitch(np.rot90(image_output, 1)), 3)
            elif glitch_direction == "horizontal":
                image_output = self.apply_glitch(image_output)
            # for 2 directional glitches, it will be either horizontal or vertical direction first
            else:
                if random.random() > 0.5:
                    image_output = self.apply_glitch(image_output)
                    image_output = np.rot90(self.apply_glitch(np.rot90(image_output, 1)), 3)
                else:
                    image_output = np.rot90(self.apply_glitch(np.rot90(image_output, 1)), 3)
                    image_output = self.apply_glitch(image_output)

            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGRA2GRAY)

            return image_output

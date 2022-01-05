import random

import cv2
import numpy as np
from PIL import Image

from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class DirtyDrum(Augmentation):
    """Emulates dirty drum effect by creating stripes of vertical and
    horizontal noises.

    :param line_width_range: Pair of ints determining the range from which the
           width of a dirty drum line is sampled.
    :type line_width_range: tuple, optional
    :param direction: Direction of effect, 0=horizontal, 1=vertical, 2=both.
    :type direction: int, optional
    :param noise_intensity: Intensity of dirty drum effect, recommended value
           range from 0.8 to 1.0.
    :type noise_intensity: float, optional
    :param ksize: Tuple of height/width pairs from which to sample the kernel
           size. Higher value increases the spreadness of stripes.
    :type ksizes: tuple, optional
    :param sigmaX: Standard deviation of the kernel along the x-axis.
    :type sigmaX: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        line_width_range=(2, 8),
        direction=random.randint(0, 2),
        noise_intensity=0.95,
        ksize=(3, 3),
        sigmaX=0,
        p=1,
    ):
        super().__init__(p=p)
        self.line_width_range = line_width_range
        self.direction = direction
        self.noise_intensity = noise_intensity
        self.ksize = ksize
        self.sigmaX = sigmaX

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DirtyDrum(line_width_range={self.line_width_range}, direction={self.direction}, noise_intensity={self.noise_intensity}, ksize={self.ksize}, sigmaX={self.sigmaX},p={self.p})"

    # Blend images to produce DirtyDrum effect
    def blend(self, img, img_dirty):
        ob = OverlayBuilder("mix", img_dirty.astype("uint8"), img, 1, (1, 1), "center", 0)
        return ob.build_overlay()

    # Add noise to stripe of image
    def add_noise(self, img, side, p=0.1):
        # side = flag to put more noise at certain side
        #   0  = left side
        #   1  = right side

        # get image dimension
        ysize, xsize = img.shape[:2]

        for y in range(ysize):
            for x in range(xsize):

                if side:  # more noise on right side
                    p_score = (((x) / (xsize)) ** 0.5) * p  # non linear score with power
                else:  # more noise on left side
                    p_score = (((xsize - x) / (xsize)) ** 0.5) * p  # non linear score with power

                if p_score > random.random():
                    img[y, x] = random.randint(0, 50)
        return img

    # Create mask for drity drum effect
    def create_dirty_mask(self, img, line_width_range=(6, 18), axis=1):
        # axis 0 = horizonal stripes
        # axis 1 = vertical stripes

        img_dirty = np.ones_like(img).astype("uint8") * 255  # initialization
        ysize, xsize = img.shape[:2]

        if axis == 1:  # vertical stripes

            x = 0
            # generate initial random strip width
            current_width = random.randint(line_width_range[0], line_width_range[1]) * random.randint(1, 5)
            # flag to break
            f_break = 0

            while True:
                # create random space between lines
                if random.random() > 0.5:
                    # coordinates of stripe
                    ys = 0
                    ye = ysize
                    xs = x
                    xe = x + current_width
                    xmid = xs + int(np.floor(current_width / 2))

                    # get left portion of stripe and apply noise
                    self.add_noise(img_dirty[ys:ye, xs:xmid], 1, p=self.noise_intensity)

                    # get right portion of stripe and apply noise
                    self.add_noise(img_dirty[ys:ye, xmid:xe], 0, p=self.noise_intensity)

                # increment on next x start loc
                x += current_width

                # generate next random strip width
                current_width = (
                    random.randint(
                        line_width_range[0],
                        line_width_range[1],
                    )
                    * random.randint(1, 5)
                )

                # if next strip > image width, set it to fit into image width
                if x + (current_width) > xsize - 1:
                    current_width = xsize - 1 - x
                    if f_break:
                        break
                    else:
                        f_break = 1

        else:  # horizontal stripes

            y = 0
            # generate initial random strip height
            current_height = random.randint(line_width_range[0], line_width_range[1]) * random.randint(1, 5)
            # flag to break
            f_break = 0

            while True:
                # create random space between lines
                if random.random() > 0.5:
                    # coordinates of stripe
                    ys = y
                    ye = y + current_height
                    xs = 0
                    xe = xsize
                    ymid = ys + int(np.floor(current_height / 2))

                    # rotate, and then get left portion of stripe and apply noise
                    img_dirty_left = self.add_noise(
                        np.rot90(img_dirty[ys:ymid, xs:xe], k=1),
                        1,
                        p=self.noise_intensity,
                    )
                    # rotate back
                    img_dirty[ys:ymid, xs:xe] = np.rot90(img_dirty_left, k=3)

                    # rotate, and get right portion of stripe and apply noise
                    img_dirty_right = self.add_noise(
                        np.rot90(img_dirty[ymid:ye, xs:xe], k=1),
                        0,
                        p=self.noise_intensity,
                    )
                    # rotate back
                    img_dirty[ymid:ye, xs:xe] = np.rot90(img_dirty_right, k=3)

                # increment on next y start loc
                y += current_height
                # generate random strip width
                current_height = (
                    random.randint(
                        line_width_range[0],
                        line_width_range[1],
                    )
                    * random.randint(1, 5)
                )

                # if next strip > image height, set it to fit into image height
                if y + (current_height) > ysize - 1:
                    current_height = ysize - 1 - y
                    if f_break:
                        break
                    else:
                        f_break = 1

        return img_dirty

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            if self.direction == 0:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 0)
                # Apply gaussian blur to mask of dirty drum
                image_dirty = cv2.GaussianBlur(image_dirty, ksize=self.ksize, sigmaX=self.sigmaX)
            elif self.direction == 1:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                image_dirty = cv2.GaussianBlur(image_dirty, ksize=self.ksize, sigmaX=self.sigmaX)
            else:
                # Create directional masks for dirty drum effect
                image_dirty_h = self.create_dirty_mask(image, self.line_width_range, 0)
                image_dirty_v = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                image_dirty_h = cv2.GaussianBlur(image_dirty_h, ksize=self.ksize, sigmaX=self.sigmaX)
                image_dirty_v = cv2.GaussianBlur(image_dirty_v, ksize=self.ksize, sigmaX=self.sigmaX)
                # Blend image with the masks of dirty drum effect
                image_dirty = self.blend(image_dirty_v, image_dirty_h)

            image_dirty_drum = self.blend(image, image_dirty)

            return image_dirty_drum

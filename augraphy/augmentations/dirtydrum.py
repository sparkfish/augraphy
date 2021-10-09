import random

import cv2
import numpy as np
from PIL import Image

from augraphy.base.augmentation import Augmentation


class DirtyDrum(Augmentation):
    """Emulates dirty drum effect by creating stripes of vertical and
    horizontal noises.

    :param line_width_range: Pair of ints determining the range from which the
           width of a dirty drum line is sampled.
    :type line_width_range: tuple, optional
    :param ksize: Tuple of height/width pairs from which to sample the kernel
           size. Higher value increases the spreadness of stripes.
    :type ksizes: tuple, optional
    :param sigmaX: Standard deviation of the kernel along the x-axis.
    :type sigmaX: float, optional
    :param alpha: Intensity of dirty drum effect, recommended value range from
            0.5 to 0.7.
    :type alpha: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        line_width_range=(6, 18),
        ksize=(17, 17),
        sigmaX=0,
        alpha=0.5,
        p=1,
    ):
        super().__init__(p=p)
        self.line_width_range = line_width_range
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.alpha = alpha

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DirtyDrum(line_width_range={self.line_width_range}, ksize={self.ksize}, sigmaX={self.sigmaX},alpha={self.alpha},p={self.p})"

    # Blend images to produce DirtyDrum effect
    def blend(self, img, img_bleed, alpha):
        img_PIL = Image.fromarray(img)
        img_bleed_PIL = Image.fromarray(img_bleed)
        img_PIL = img_PIL.convert("RGBA")
        img_bleed_PIL = img_bleed_PIL.convert("RGBA")
        img_blended = Image.blend(img_PIL, img_bleed_PIL, alpha=alpha)
        return np.array(img_blended)

    # Add noise to stripe of image
    def add_noise(self, img, side, p=0.1):
        # side = flag to put more noise at certain side
        #   0  = left side
        #   1  = right side

        # get image dimension
        if len(img.shape) > 2:
            ysize, xsize, dim = img.shape
        else:
            ysize, xsize = img.shape
            dim = 2

        for y in range(ysize):
            for x in range(xsize):

                if side:  # more noise on right side
                    p_score = (((x) / xsize) ** 0.5) * p  # non linear score with power
                else:  # more noise on left side
                    p_score = (((xsize - x) / xsize) ** 0.5) * p  # non linear score with power

                if p_score > random.random():
                    img[y, x] = 0
        return img

    # Create mask for drity drum effect
    def create_dirty_mask(self, img, line_width_range=(6, 18), axis=1):
        # axis 0 = horizonal stripes
        # axis 1 = vertical stripes

        img_dirty = np.ones_like(img).astype("uint8") * 255  # initialization

        if len(img.shape) > 2:
            ysize, xsize, dim = img.shape
        else:
            ysize, xsize = img.shape

        if axis == 1:  # vertical stripes
            x = 0
            # generate initial random strip width
            current_width2 = random.randint(line_width_range[0], line_width_range[1])
            current_width = current_width2 * random.randint(1, 5)
            while x + (current_width + current_width2) < xsize:  # while next stripe is smaller than size

                # coordinates of stripe
                ys = 0
                ye = ysize
                xs = x
                xe = x + current_width
                xmid = xs + int(np.floor(current_width / 2))

                # get left portion of stripe and apply noise
                img_dirty_left = self.add_noise(img_dirty[ys:ye, xs:xmid], 0, p=0.5)
                img_dirty[ys:ye, xs:xmid] = img_dirty_left

                # get right portion of stripe and apply noise
                img_dirty_right = self.add_noise(img_dirty[ys:ye, xmid:xe], 1, p=0.5)
                img_dirty[ys:ye, xmid:xe] = img_dirty_right

                # alternate stripe in opposite intensity (black -> white or white -> black)
                img_dirty[ys:ye, xe : xe + current_width2] = random.randint(80, 160)

                # increment on next x start loc
                x += current_width + current_width2
                # generate random strip width
                current_width2 = random.choice(line_width_range)
                current_width = current_width2 * random.randint(1, 5)

        else:  # horizontal stripes
            y = 0
            # generate initial random strip height

            current_height2 = random.randint(line_width_range[0], line_width_range[1])
            current_height = current_height2 * random.randint(1, 5)
            while y + (current_height) < ysize:  # while next stripe is smaller than size

                # coordinates of stripe
                ys = y
                ye = y + current_height
                xs = 0
                xe = xsize
                ymid = ys + int(np.floor(current_height / 2))

                # rotate, and then get left portion of stripe and apply noise
                img_dirty_left = self.add_noise(
                    np.rot90(img_dirty[ys:ymid, xs:xe], k=1),
                    0,
                    p=0.5,
                )
                img_dirty[ys:ymid, xs:xe] = np.rot90(img_dirty_left, k=3)  # rotate back

                # rotate, and get right portion of stripe and apply noise
                img_dirty_right = self.add_noise(
                    np.rot90(img_dirty[ymid:ye, xs:xe], k=1),
                    1,
                    p=0.5,
                )
                img_dirty[ymid:ye, xs:xe] = np.rot90(
                    img_dirty_right,
                    k=3,
                )  # rotate back

                # alternate stripe in opposite intensity (black -> white or white -> black)
                img_dirty[ye : ye + current_height2, xs:xe] = random.randint(80, 160)

                # increment on next x start loc
                y += current_height + current_height2
                # generate random strip width
                current_height2 = random.randint(
                    line_width_range[0],
                    line_width_range[1],
                )
                current_height = current_height2 * random.randint(1, 5)

        return img_dirty

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            direction = random.choice([0, 1, 2])

            if direction == 0:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 0)
                # Apply gaussian blur to mask of dirty drum
                # image_dirty = cv2.GaussianBlur(image_dirty, ksize=self.ksize, sigmaX=self.sigmaX)
            elif direction == 1:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                # image_dirty = cv2.GaussianBlur(image_dirty, ksize=self.ksize, sigmaX=self.sigmaX)
            else:
                # Create directional masks for dirty drum effect
                image_dirty_h = self.create_dirty_mask(image, self.line_width_range, 0)
                image_dirty_v = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                # image_dirty_h = cv2.GaussianBlur(image_dirty_h, ksize=self.ksize, sigmaX=self.sigmaX)
                # image_dirty_v = cv2.GaussianBlur(image_dirty_v, ksize=self.ksize, sigmaX=self.sigmaX)
                # Blend image with the masks of dirty drum effect
                image_dirty = self.blend(image_dirty_v, image_dirty_h, 0.5)

            image_dirty_drum = self.blend(image, image_dirty, self.alpha)

            return image_dirty_drum

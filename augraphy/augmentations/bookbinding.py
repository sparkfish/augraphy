import math
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class BookBinding(Augmentation):
    """Creates a book binding effect with shadow and curved lines

    :param radius_range: The range of radius in pixels
    :type range: tuple, optional
    :param curve_intensity_range: Intensity by which the page text should be curved
    :type range: tuple, optional
    :param mirror: Flag to enable mirror effect.
    :type mirror: int, optional
    :param mirror_range: Tuple of floats to determine width of image to be mirrored.
    :type mirror_range: Tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        radius_range=(1, 100),
        curve_intensity_range=(0, 70),
        mirror=0,
        mirror_range=(0.1, 0.2),
        p=1,
    ):
        super().__init__(p=p)
        self.radius_range = radius_range
        self.curve_intensity_range = curve_intensity_range
        self.mirror = mirror
        self.mirror_range = mirror_range

    def __repr__(self):
        return f"BookBinding(radius_range={self.radius_range}, curve_intensity_range={self.curve_intensity_range}, mirror={self.mirror}, mirror_range={self.mirror_range},  p={self.p})"

    def add_book_shadow(self, img, radius, angle):

        rows = img.shape[0]
        cols = img.shape[1]

        # compute mask of shadow
        img_dist = np.repeat(np.arange(cols), rows)
        img_dist = np.transpose(img_dist.reshape(cols, rows))
        img_d = img_dist + (radius * (1 - math.cos(angle)))
        img_mask = (img_dist / img_d) ** 2
        # rescale 0- 1 to prevent darken of the image
        img_mask = (img_mask - np.min(img_mask)) / (np.max(img_mask) - np.min(img_mask))

        # overlay mask of shadow to input image
        ob = OverlayBuilder("darken", (img_mask * 255).astype("uint8"), img, 1, (1, 1), "center", 0, 1)
        img_output = ob.build_overlay()

        return img_output.astype("uint8")

    def curve_page(self, img, curve_intensity):
        rows = img.shape[0]
        cols = img.shape[1]
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = 0
                offset_y = int(curve_intensity * math.sin(2 * 3.14 * j / (3 * rows)))
                if i + offset_y < rows:
                    img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
                else:
                    img_output[i, j] = img[0, 0]
        return img_output

    def __call__(self, image, layer=None, force=False):
        image = image.copy()
        radius = random.randint(self.radius_range[0], self.radius_range[1])
        angle = 30
        curve_intensity = random.randint(
            self.curve_intensity_range[0],
            self.curve_intensity_range[1],
        )
        image = self.add_book_shadow(image, radius, angle)
        image = self.curve_page(image, curve_intensity)

        if self.mirror:
            # get image size and channels
            if len(image.shape) > 2:
                ysize, xsize, channels = image.shape
            else:
                ysize, xsize = image.shape
                channels = 1

            # generate range of mirror
            mirror_range = np.random.uniform(self.mirror_range[0], self.mirror_range[1])
            xsize_mirror = int(mirror_range * xsize)
            image_mirror = image[:, :xsize_mirror]

            # create new image with original size + mirror size
            if channels > 2:
                new_image = np.zeros((ysize, xsize + xsize_mirror, channels)).astype("uint8")
            else:
                new_image = np.zeros((ysize, xsize + xsize_mirror)).astype("uint8")

            # merged the original iamge + mirrored image
            new_image[:, :xsize_mirror] = np.fliplr(image_mirror)
            new_image[:, xsize_mirror:] = image

            image = new_image

        return image

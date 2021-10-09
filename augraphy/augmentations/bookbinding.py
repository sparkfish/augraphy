import math
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class BookBinding(Augmentation):
    """Creates a book binding effect with shadow and curved lines

    :param radius_range: The range of radius in pixels
    :type range: tuple, optional
    :param curve_intensity_range: Intensity by which the page text should be curved
    :type range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        radius_range=(1, 100),
        curve_intensity_range=(0, 70),
        p=1,
    ):
        super().__init__(p=p)
        self.radius_range = radius_range
        self.curve_intensity_range = curve_intensity_range

    def __repr__(self):
        return f"BookBinding(radius_range={self.radius_range}, curve_intensity_range={self.curve_intensity_range},  p={self.p})"

    def add_book_shadow(self, img, radius, angle):
        img_output = img.copy()
        rows = img.shape[0]
        cols = img.shape[1]
        for i in range(rows):
            for j in range(cols):
                dist = math.sqrt((j * j))
                d = dist + (radius * (1 - math.cos(angle)))
                new_i = img[i, j] * (dist / (d)) ** 2
                img_output[i, j, :] = new_i
        return img_output

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
        return image

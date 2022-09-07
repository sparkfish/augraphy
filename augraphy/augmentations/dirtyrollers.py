import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class DirtyRollers(Augmentation):
    """Emulates an effect created by certain document scanners.

    :param line_width_range: Pair of ints determining the range from which the
    width of a dirty roller line is sampled.
    :type line_width_range: tuple, optional
    :param scanline_type: Types of scanline, use 0 for white background.
    :type scanline_type: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        line_width_range=(8, 12),
        scanline_type=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.line_width_range = line_width_range
        self.scanline_type = scanline_type

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DirtyRollers(line_width_range={self.line_width_range}, scanline_type={self.scanline_type}, p={self.p})"

    def apply_scanline_mask(self, img, mask, meta_mask):
        """Main function to apply scanline mask to input image.
        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        :param meta_mask: Meta mask of scanline effect.
        :type meta_mask: numpy.array (numpy.uint8)
        """

        # for dark background
        if self.scanline_type:
            return self.apply_scanline_mask_v2(img, mask, meta_mask)
        # for white background
        else:
            return self.apply_scanline_mask_v1(img, mask, meta_mask)

    def apply_scanline_mask_v2(self, img, mask, meta_mask):
        """Function to apply scanline mask to input image with dark background.
        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        :param meta_mask: Meta mask of scanline effect.
        :type meta_mask: numpy.array (numpy.uint8)
        """
        mask = self.apply_scanline_metamask_v2(mask, meta_mask)
        update_lambda = lambda x, y: min(255, x + (x * (1 - (y / 100))))
        update = np.vectorize(update_lambda)
        return update(img, mask)

    def apply_scanline_metamask_v2(self, img, mask):
        """Function to apply scanline meta mask to scanline mask of dark background.
        :param img: The mask image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Meta mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        """
        update_lambda = lambda x, y: max(0, x - (x * (1 - (y / 100))))
        update = np.vectorize(update_lambda)
        return update(img, mask)

    def apply_scanline_mask_v1(self, img, mask, meta_mask):
        """Function to apply scanline mask to input image with white background.
        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        :param meta_mask: Meta mask of scanline effect.
        :type meta_mask: numpy.array (numpy.uint8)
        """
        mask = self.apply_scanline_metamask_v1(mask, meta_mask)
        update_lambda = lambda x, y: max(0, x - (x * (1 - (y / 100))))
        update = np.vectorize(update_lambda)
        return update(img, mask)

    def apply_scanline_metamask_v1(self, img, mask):
        """Function to apply scanline meta mask to scanline mask of white background.
        :param img: The mask image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Meta mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        """
        update_lambda = lambda x, y: min(99, x + (x * (1 - (y / 100))))
        update = np.vectorize(update_lambda)
        return update(img, mask)

    def create_scanline_mask(self, width, height, line_width):
        """Function to create scanline mask.
        :param width: Width of scanline mask.
        :type width: int
        :param height: Height of scanline mask.
        :type height: int
        :param line_width: Width of a dirty roller line.
        :type line_width: int
        """
        grad_list = list()

        # Create Standard Bar
        grad_high_pct = random.randint(86, 99)
        grad_low_pct = random.randint(70, 85)

        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        # Create Standard Bar with Wide Dark
        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_center = np.full((random.randint(1, 6)), grad_low_pct)
        grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        # Create Standard Bar with Wide Light
        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
        grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        # Create Standard Bar with Lower Dark
        grad_high_pct += min(100, random.randint(-3, 3))
        grad_low_pct -= random.randint(5, 8)
        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        # Create Standard Bar with Low Dark and Wide Dark
        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_center = np.full((random.randint(1, 6)), grad_low_pct)
        grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        # Create Standard Bar with Low Dark Wide Light
        grad_dec = (grad_low_pct - grad_high_pct) / (line_width)
        grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
        grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
        grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
        grad_grid = np.tile(grad_grid, (height, 1))
        grad_list.append(grad_grid)

        mask = random.choice(grad_list)
        while mask.shape[1] < width:
            mask = np.hstack((mask, random.choice(grad_list)))

        return mask[:, 0:width]

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            line_width = random.randint(
                self.line_width_range[0],
                self.line_width_range[1],
            )
            rotate = random.choice([True, False])

            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mask = self.create_scanline_mask(image.shape[1], image.shape[0], line_width)

            meta_mask = self.create_scanline_mask(
                image.shape[1],
                image.shape[0],
                line_width * random.randint(10, 25),
            )
            image = self.apply_scanline_mask(image, mask, meta_mask).astype("uint8")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            return image

import math
import random

import numpy as np

from augraphy.augmentations.lib import four_point_transform
from augraphy.augmentations.pageborder import PageBorder
from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class BookBinding(Augmentation):
    """Creates a book binding effect with shadow and curved lines

    :param radius_range: The range of radius in pixels.
    :type radius_range: tuple, optional
    :param curve_range: Pixels by which the page text should be curved, suggested value is 1/8 of image width.
    :type curve_range: tuple, optional
    :param mirror_range: Tuple of floats to determine percentage of image to be mirrored.
    :type mirror_range: Tuple, optional
    :param curling_direction: The direction of page curling, 0: up, 1: down.
    :type curling_direction: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        radius_range=(1, 100),
        curve_range=(200, 300),
        mirror_range=(0.2, 0.5),
        curling_direction=0,
        p=1,
    ):
        super().__init__(p=p)
        self.radius_range = radius_range
        self.curve_range = curve_range
        self.mirror_range = mirror_range
        self.curling_direction = curling_direction

    def __repr__(self):
        return f"BookBinding(radius_range={self.radius_range}, curve_range={self.curve_range}, mirror_range={self.mirror_range}, curling_direction={self.curling_direction}, p={self.p})"

    def add_book_shadow(self, img, radius, angle):
        """Add shadow effect in the input image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param radius: Radius of the shadow effect.
        :type radius: int
        :param angle: Angle value to generate shadow effect.
        :type angle: int
        """

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
        ob = OverlayBuilder(
            "darken",
            (img_mask * 255).astype("uint8"),
            img,
            1,
            (1, 1),
            "center",
            0,
            1,
        )
        img_output = ob.build_overlay()

        return img_output.astype("uint8")

    def curve_page(self, img, curve_range):
        """Generate curvy effect in the input image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param curve_range: Tuple of pixels by which the page text should be curved.
        :type curve_range: tuple

        """

        rows = img.shape[0]
        cols = img.shape[1]

        if curve_range > cols / 2:
            curve_range = int(cols / 2)

        # x coordinate when offset_y starts to become negative
        x_negative = -1

        # reassign variable name for clarity
        max_offset_y = curve_range

        if len(img.shape) > 2:
            channels = img.shape[2]
            img_output = np.zeros(
                (rows + max_offset_y, cols, channels),
                dtype=img.dtype,
            )
        else:
            img_output = np.zeros((rows + max_offset_y, cols), dtype=img.dtype)

        for y in range(rows):
            y_new = y + max_offset_y
            for x in range(cols):

                offset_x = 0
                offset_y = int(curve_range * math.sin(2 * 3.14 * x / (3 * rows)))

                # for negative offset_y
                if offset_y < 0:
                    # set value of x when offset_y turns negative
                    if x_negative == -1:
                        x_negative = x

                    offseted_y = y - offset_y
                    offseted_x = x - offset_x
                    check_offset = y - offset_y
                # for positive offset_y
                else:
                    offseted_y = y + offset_y
                    offseted_x = x + offset_x
                    check_offset = y + offset_y

                # assign new y value
                if check_offset < rows:
                    img_output[y_new, x] = img[
                        (offseted_y) % (rows),
                        (offseted_x) % cols,
                    ]
                else:
                    # img_output[y_new, x] = 0
                    # add top section
                    img_output[(y_new - rows), x] = img[
                        (offseted_y) % (rows),
                        (offseted_x) % cols,
                    ]

        # remove negative offset part
        if x_negative != -1:
            img_output = img_output[:, :x_negative]

        return img_output

    def __call__(self, image, layer=None, force=False):
        image = image.copy()
        radius = random.randint(self.radius_range[0], self.radius_range[1])
        angle = 30
        curve_range = max(
            int(image.shape[1] / 10),
            random.randint(
                self.curve_range[0],
                self.curve_range[1],
            ),
        )

        # get bending direction
        curve_down = self.curling_direction

        added_border_height = int(image.shape[0] / 10)

        # right page - add page border, add shadow and then bend page
        if curve_down:
            page_border = PageBorder(
                side="top",
                border_background_value=(100, 120),
                flip_border=0,
                width_range=(added_border_height, added_border_height),
                pages=random.randint(6, 8),
                noise_intensity_range=(0.05, 0.1),
                curve_frequency=(1, 3),
                curve_height=(1, 3),
                curve_length_one_side=(10, 10),
                value=(30, 60),
                same_page_border=0,
                p=1,
            )

            image_added_border_right = page_border(image)
            image_shadow_right = self.add_book_shadow(np.flipud(image_added_border_right), radius, angle)
            image_right = np.flipud(self.curve_page(image_shadow_right, curve_range))
        else:
            page_border = PageBorder(
                side="bottom",
                border_background_value=(100, 120),
                flip_border=1,
                width_range=(added_border_height, added_border_height),
                pages=random.randint(6, 8),
                noise_intensity_range=(0.05, 0.1),
                curve_frequency=(1, 3),
                curve_height=(1, 3),
                curve_length_one_side=(10, 10),
                value=(30, 60),
                same_page_border=0,
                p=1,
            )

            image_added_border_right = page_border(image)
            image_shadow_right = self.add_book_shadow(image_added_border_right, radius, angle)
            image_right = self.curve_page(image_shadow_right, curve_range)

        # left page - add page border, add shadow and then bend page
        if curve_down:
            page_border = PageBorder(
                side="top",
                border_background_value=(100, 120),
                flip_border=1,
                width_range=(added_border_height, added_border_height),
                pages=random.randint(6, 8),
                noise_intensity_range=(0.05, 0.1),
                curve_frequency=(1, 3),
                curve_height=(1, 3),
                curve_length_one_side=(10, 10),
                value=(30, 60),
                same_page_border=0,
                p=1,
            )

            image_added_border_left = page_border(image)
            image_shadow_left = np.fliplr(self.add_book_shadow(np.fliplr(image_added_border_left), radius, angle))
            image_left = self.curve_page(np.fliplr(np.flipud(image_shadow_left)), curve_range * 3)
        else:
            page_border = PageBorder(
                side="top",
                border_background_value=(100, 120),
                flip_border=0,
                width_range=(added_border_height, added_border_height),
                pages=random.randint(6, 8),
                noise_intensity_range=(0.05, 0.1),
                curve_frequency=(1, 3),
                curve_height=(1, 3),
                curve_length_one_side=(10, 10),
                value=(30, 60),
                same_page_border=0,
                p=1,
            )

            image_added_border_left = page_border(np.fliplr(np.flipud(image)))
            image_shadow_left = np.fliplr(self.add_book_shadow(image_added_border_left, radius, angle))
            if image.shape[1] > image.shape[0]:
                left_curve_range = curve_range
            else:
                left_curve_range = curve_range * 3
            image_left = self.curve_page(np.flipud(image_shadow_left), left_curve_range)

        # further bend left image by using perspective transform
        ysize, xsize = image_left.shape[:2]
        curve_xsize = random.randint(int(xsize / 10), int(xsize / 8))

        # bending size
        if curve_down:
            curve_xsize = int(xsize / 2)
        else:
            curve_xsize = int(xsize / 2)

        # right image x and y size
        cysize, cxsize = image_right.shape[:2]

        # source and target points of the bending process
        source_points = np.float32([[0, 0], [cxsize, 0], [cxsize, cysize], [0, cysize]])
        target_points = np.float32([[curve_xsize, 0], [cxsize, 0], [cxsize, cysize], [curve_xsize, cysize]])

        if curve_down:
            img_left = np.fliplr(np.flipud(image_left.astype("float"))) / 255
        else:
            img_left = image_left.astype("float") / 255

        # get bended image
        image_left = (four_point_transform(img_left, source_points, target_points, cxsize, cysize) * 255).astype(
            "uint8",
        )
        # remove the empty section after the transform
        image_left = image_left[:, curve_xsize:]
        # generate range of mirror and crop image based on mirror size
        mirror_range = np.random.uniform(self.mirror_range[0], self.mirror_range[1])
        image_left = image_left[:, image_left.shape[1] - int(image_left.shape[1] * mirror_range) :]
        # get new y and x size of left image
        ysize, xsize = image_left.shape[:2]

        # get their y difference
        y_diff = cysize - ysize

        if not curve_down:
            image_left = np.fliplr(self.add_book_shadow(np.fliplr(image_left), radius, angle))

        # create new image with original size + mirror size
        if len(image_right.shape) > 2:
            new_image = np.zeros((ysize + y_diff, xsize + cxsize, image_right.shape[2])).astype(
                "uint8",
            )
        else:
            new_image = np.zeros((ysize + y_diff, xsize + cxsize)).astype("uint8")

        # merged left image and right image
        if curve_down:
            new_image[:, :xsize] = image_left
        else:
            if image.shape[0] < image.shape[1]:
                start_y = 0
                new_image[start_y : image_left.shape[0], :xsize] = image_left
            else:
                start_y = curve_range
                random_offset = random.randint(int(ysize / 30), int(ysize / 25))
                random_offset = 0
                new_image[start_y - random_offset :, :xsize] = image_left[: -start_y + random_offset, :]
        if y_diff != 0:
            new_image[:-y_diff, xsize:] = image_right
        else:
            new_image[:, xsize:] = image_right

        image_out = new_image

        return image_out

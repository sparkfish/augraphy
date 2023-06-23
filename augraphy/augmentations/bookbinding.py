import math
import random

import cv2
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import four_point_transform
from augraphy.augmentations.lib import load_image_from_cache
from augraphy.augmentations.pageborder import PageBorder
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class BookBinding(Augmentation):
    """Creates a book binding effect with shadow and curved lines

    :param radius_range: The range of radius in pixels for the darken shadow effect.
    :type radius_range: tuple, optional
    :param curve_range: Pixels by which the page text should be curved, suggested value is 1/8 of image width.
    :type curve_range: tuple, optional
    :param mirror_range: Tuple of floats to determine percentage of image to be mirrored.
    :type mirror_range: Tuple, optional
    :param curling_direction: The direction of page curling, -1: random, 0: up, 1: down.
    :type curling_direction: int, optional
    :param backdrop_color: The backdrop color (BGR) of the book binding effect.
    :type backdrop_color: tuple, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        radius_range=(1, 100),
        curve_range=(200, 300),
        mirror_range=(1.0, 1.0),
        curling_direction=-1,
        backdrop_color=(0, 0, 0),
        numba_jit=1,
        p=1,
    ):
        super().__init__(p=p, numba_jit=numba_jit)
        self.radius_range = radius_range
        self.curve_range = curve_range
        self.mirror_range = mirror_range
        self.curling_direction = curling_direction
        self.backdrop_color = backdrop_color
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        return f"BookBinding(radius_range={self.radius_range}, curve_range={self.curve_range}, mirror_range={self.mirror_range}, curling_direction={self.curling_direction}, backdrop_color={self.backdrop_color}, numba_jit={self.numba_jit}, p={self.p})"

    def add_book_shadow(self, img, radius, angle=30):
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

        min_value = np.min(img_mask)
        max_value = np.max(img_mask)

        # rescale mask to 0.2 - 1 (prevent total black area)
        min_intensity = 0.2
        max_intensity = 1.0
        img_mask = ((img_mask - min_value) / (max_value - min_value)) * (max_intensity - min_intensity) + min_intensity

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

    def curve_page(self, img, curve_value):
        """Generate curvy effect in the input image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param curve_value: Pixel number of the page text should be curved.
        :type curve_value: int

        """
        rows = img.shape[0]
        cols = img.shape[1]

        # curve value not more than image height
        curve_value = min(rows, curve_value)

        if len(img.shape) > 2:
            channels = img.shape[2]
            img_output = np.full(
                (rows + curve_value, cols, channels),
                fill_value=self.backdrop_color,
                dtype=img.dtype,
            )

        else:
            img_output = np.full((rows + curve_value, cols), fill_value=np.mean(self.backdrop_color), dtype=img.dtype)

        curve_factor = (cols * 2) + (cols * 2 * curve_value / rows)
        self.curve_page_processing(img, img_output, curve_value, rows, cols, curve_factor)

        return img_output

    @staticmethod
    @jit(nopython=True, cache=True)
    def curve_page_processing(img, img_output, curve_value, rows, cols, curve_factor):
        """Function to apply sin function to create wavy image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param img_output: The output image from the function.
        :type img_output: numpy.array (numpy.uint8)
        :param curve_value: Pixel number of the page text should be curved.
        :type curve_value: int
        :param rows: Number of rows in input image.
        :type rows: int
        :param cols: Number of columns in input image.
        :type cols: int
        :param curve_factor: The curve factor determines the opening size of book binding effect
        :type curve_factor: float
        """
        for y in range(rows):
            for x in range(cols):
                offset_y = int(curve_value * math.sin(2 * 3.14 * x / curve_factor))
                offseted_y = y + offset_y
                img_output[offseted_y, x] = img[y, x]

        return img_output

    def curve_processing(self, image, image_left):
        """Core function for curvy effect processing.

        :param image: The right image of the book binding effect.
        :type image: numpy.array (numpy.uint8)
        :param image_left: The left image of the book binding effect.
        :type image_left: numpy.array (numpy.uint8)
        """

        if self.curling_direction == -1 or self.curling_direction == "random":
            curve_down = random.choice([0, 1])
        else:
            curve_down = self.curling_direction

        # generate parameters
        radius = random.randint(self.radius_range[0], self.radius_range[1])
        curve_value = random.randint(self.curve_range[0], self.curve_range[1])
        page_rotation = random.randint(3, 3)
        page_number = random.randint(8, 12)
        curve_frequency = (1, 3)
        curve_height = (1, 3)
        curve_length_one_side = (10, 30)

        # from PageBorder, min value is 1
        self.backdrop_color = (
            max(1, self.backdrop_color[0]),
            max(1, self.backdrop_color[1]),
            max(1, self.backdrop_color[2]),
        )

        # page borders
        added_border_height = int(max(image.shape[:2]) / 20)

        # right side of image
        # create borders
        page_border = PageBorder(
            page_border_width_height=(int(added_border_height / 2), -added_border_height),
            page_border_color=(0, 0, 0),
            page_border_background_color=self.backdrop_color,
            page_numbers=random.randint(page_number, page_number),
            page_rotation_angle_range=(page_rotation, page_rotation),
            curve_frequency=curve_frequency,
            curve_height=curve_height,
            curve_length_one_side=curve_length_one_side,
            same_page_border=0,
            numba_jit=1,
            p=1,
        )

        if not curve_down:
            image = np.flipud(image)

        image_shadow = self.add_book_shadow(image, radius)
        image_added_border_right = page_border(np.rot90(image_shadow, 3))
        image_added_border_right = np.rot90(image_added_border_right[added_border_height:, :])
        image_right = self.curve_page(image_added_border_right, curve_value)

        if not curve_down:
            image_right = np.flipud(image_right)

        # left side of image
        # create borders
        page_border = PageBorder(
            page_border_width_height=(int(-added_border_height / 2), -added_border_height),
            page_border_color=(0, 0, 0),
            page_border_background_color=self.backdrop_color,
            page_numbers=random.randint(page_number, page_number),
            page_rotation_angle_range=(page_rotation, page_rotation),
            curve_frequency=curve_frequency,
            curve_height=curve_height,
            curve_length_one_side=curve_length_one_side,
            same_page_border=0,
            numba_jit=1,
            p=1,
        )

        if not curve_down:
            image_left = np.flipud(image_left)

        image_shadow_left = self.add_book_shadow(np.fliplr(image_left), radius)
        image_added_border_left = np.fliplr(page_border(image_shadow_left))
        image_added_border_left = image_added_border_left[:, : -int(added_border_height / 2)]
        image_left = np.fliplr(self.curve_page(np.fliplr(image_added_border_left), curve_value))

        if not curve_down:
            image_left = np.flipud(image_left)

        # further bend left image by using perspective transform
        ysize, xsize = image_left.shape[:2]

        # generate range of mirror and crop image based on mirror size
        mirror_range = np.random.uniform(self.mirror_range[0], self.mirror_range[1])
        image_left = image_left[:, image_left.shape[1] - int(image_left.shape[1] * mirror_range) :]

        # get new y and x size of left image
        ysize, xsize = image_left.shape[:2]
        cysize, cxsize = image_right.shape[:2]

        # create new image with original size + mirror size
        if len(image.shape) > 2:
            image_output = np.zeros((max(ysize, cysize), xsize + cxsize, image.shape[2]), dtype="uint8")
        else:
            image_output = np.zeros((max(ysize, cysize), xsize + cxsize), dtype="uint8")

        # merged left image and right image
        image_output[:ysize, :xsize] = image_left
        image_output[:cysize, xsize:] = image_right

        # get mask for shadow effect
        if len(image.shape) > 2:
            backdrop_color = self.backdrop_color
        else:
            backdrop_color = np.mean(self.backdrop_color)
        image_mask = np.zeros_like(image_output, dtype="uint8")
        if not curve_down:
            image_output = np.flipud(image_output)
        for x in range(image_mask.shape[1]):
            for y in range(image_mask.shape[0]):
                if len(image.shape) > 2:
                    color_eval = all(image_output[y, x] == backdrop_color)
                else:
                    color_eval = image_output[y, x] == backdrop_color
                if color_eval:
                    image_mask[y, x] = 1
                else:
                    break

        if not curve_down:
            image_output = np.flipud(image_output)
            image_mask = np.flipud(image_mask)

        # add shadow effect
        image_darken = image_output.astype("float") * 0.7
        offset = random.randint(int(image.shape[0] / 40), int(image.shape[0] / 30))
        image_darken[image_mask > 0] = image_output[image_mask > 0]
        if curve_down:
            image_darken[:-offset, :] = image_darken[offset:, :]
        else:
            image_darken[offset:, :] = image_darken[:-offset, :]
        image_darken = cv2.GaussianBlur(image_darken.astype("uint8"), (151, 151), cv2.BORDER_DEFAULT)
        image_output[image_mask > 0] = image_darken[image_mask > 0]

        return image_output

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # load left side of image from cache
            image_left = load_image_from_cache()
            if image_left is not None:

                # resize foreground
                image_left = cv2.resize(
                    image_left,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                # consistent color channel
                if len(image.shape) > 2 and len(image_left) < 3:
                    image_left = cv2.cvtColor(image_left, cv2.COLOR_GRAY2BGR)
                elif len(image.shape) < 3 and len(image_left) > 2:
                    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)

            else:
                image_left = image.copy()

            image_output = self.curve_processing(image, image_left)

            return image_output

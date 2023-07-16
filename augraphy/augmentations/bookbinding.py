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
    """Creates a book binding effect by applying shadow and combining two bended images.
        The input image will be served as the right image.
        If cache image is set, the left image will be selected from cache images, else it will be using the input image.

    :param shadow_radius_range: The range of radius in pixels for the darken shadow effect.
    :type shadow_radius_range: tuple, optional
    :param curve_range_right: Tuple of ints determing the amount of curving down pixels in the right image.
    :type curve_range_right: tuple, optional
    :param curve_range_left: Tuple of ints determing the amount of curving down pixels in the left image.
    :type curve_range_left: tuple, optional
    :param curve_ratio_right: Tuple of floats determing the percentage of right image (in width) should be squeezed horizontally.
    :type curve_ratio_right: tuple, optional
    :param curve_ratio_left: Tuple of floats determing the percentage of left image (in width) should be squeezed horizontally.
    :type curve_ratio_left: tuple, optional
    :param mirror_range: Tuple of floats to determine percentage of image to be mirrored.
    :type mirror_range: Tuple, optional
    :param binding_align: Flag to determine whether the pages in the binding effect should be properly aligned or not.
    :type binding_align int, optional
    :param binding_pages: Tuple of ints determing the number of pages in the book binding effect.
    :type binding_pages tuple, optional
    :param curling_direction: The direction of page curling, -1: random, 0: up, 1: down.
    :type curling_direction: int, optional
    :param backdrop_color: The backdrop color (BGR) of the book binding effect.
    :type backdrop_color: tuple, optional
    :param use_cache_images: Flag to enable the usage of cache images in creating book binding effect.
    :type use_cache_images: int, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        shadow_radius_range=(30, 100),
        curve_range_right=(50, 100),
        curve_range_left=(200, 300),
        curve_ratio_right=(0.05, 0.1),
        curve_ratio_left=(0.5, 0.6),
        mirror_range=(1.0, 1.0),
        binding_align="random",
        binding_pages=(5, 10),
        curling_direction=-1,
        backdrop_color=(0, 0, 0),
        use_cache_images=1,
        numba_jit=1,
        p=1,
    ):
        super().__init__(p=p, numba_jit=numba_jit)
        self.shadow_radius_range = shadow_radius_range
        self.curve_range_right = curve_range_right
        self.curve_range_left = curve_range_left
        self.curve_ratio_right = curve_ratio_right
        self.curve_ratio_left = curve_ratio_left
        self.mirror_range = mirror_range
        self.binding_align = binding_align
        self.binding_pages = binding_pages
        self.curling_direction = curling_direction
        self.backdrop_color = backdrop_color
        self.use_cache_images = use_cache_images
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        return f"BookBinding(shadow_radius_range={self.shadow_radius_range}, curve_range_right={self.curve_range_right}, curve_range_left={self.curve_range_left}, curve_ratio_right={self.curve_ratio_right}, curve_ratio_left={self.curve_ratio_left}, mirror_range={self.mirror_range}, binding_align={self.binding_align}, binding_pages={self.binding_pages}, curling_direction={self.curling_direction}, backdrop_color={self.backdrop_color}, use_cache_images={self.use_cache_images}, numba_jit={self.numba_jit}, p={self.p})"

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

    def curve_transform(self, image, curve_ratio_range, image_direction):
        """Bend image further in horizontal direction using perspective transform.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param curve_ratio: The ratio of transformation in horizontal direction..
        :type curve_ratio: float
        :param image_direction: Flag to identify left or right side of image.
        :type image_direction: string
        """

        curve_ratio = random.uniform(curve_ratio_range[0], curve_ratio_range[1])

        ysize, xsize = image.shape[:2]

        # start and end points for left and right side of images
        if image_direction == "left":
            source_points = np.float32([[0, 0], [xsize, 0], [xsize, ysize], [0, ysize]])
            target_points = np.float32(
                [[xsize * curve_ratio, 0], [xsize, 0], [xsize, ysize], [xsize * curve_ratio, ysize]],
            )
        else:
            source_points = np.float32([[0, 0], [xsize, 0], [xsize, ysize], [0, ysize]])
            target_points = np.float32(
                [[0, 0], [xsize * (1 - curve_ratio), 0], [xsize * (1 - curve_ratio), ysize], [0, ysize]],
            )

        # perspective transform to further bend image in x direction only
        image_transformed = four_point_transform(image, source_points, target_points, xsize, ysize)

        # crop the blank area after the transform
        if image_direction == "left":
            image_transformed = image_transformed[:, int(xsize * curve_ratio) :]
        else:
            image_transformed = image_transformed[:, : int(xsize * (1 - curve_ratio))]

        return image_transformed.astype("uint8")

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
        trim_sides = (1, 0, 0, 1)
        radius = random.randint(self.shadow_radius_range[0], self.shadow_radius_range[1])
        curve_value_right = random.randint(self.curve_range_right[0], self.curve_range_right[1])
        curve_value_left = random.randint(self.curve_range_left[0], self.curve_range_left[1])
        if self.binding_align == "random":
            binding_align = random.choice([0, 1])
        else:
            binding_align = self.binding_align
        if binding_align:
            angle = random.randint(2, 5)
            page_rotation = (angle, angle)
        else:
            page_rotation = (1, 5)
        page_number = random.randint(self.binding_pages[0], self.binding_pages[0])
        curve_frequency = (0, 1)
        curve_height = (1, 3)
        curve_length_one_side = (10, 30)
        if self.use_cache_images == "random":
            use_cache_images = random.choice([0, 1])
        else:
            use_cache_images = self.use_cache_images

        # page borders value
        added_border_height = int(max(image.shape[:2]) / 20)
        width_height = (int(added_border_height / 2), -added_border_height)

        # right side of image
        # create borders
        page_border = PageBorder(
            page_border_width_height=width_height,
            page_border_color=(0, 0, 0),
            page_border_background_color=self.backdrop_color,
            page_border_use_cache_images=use_cache_images,
            page_border_trim_sides=trim_sides,
            page_numbers=random.randint(page_number, page_number),
            page_rotate_angle_in_order=binding_align,
            page_rotation_angle_range=page_rotation,
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
        image_added_border_right = page_border(image_shadow)

        # left side of image
        # create borders
        page_border = PageBorder(
            page_border_width_height=width_height,
            page_border_color=(0, 0, 0),
            page_border_background_color=self.backdrop_color,
            page_border_use_cache_images=use_cache_images,
            page_border_trim_sides=trim_sides,
            page_numbers=random.randint(page_number, page_number),
            page_rotate_angle_in_order=binding_align,
            page_rotation_angle_range=page_rotation,
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

        # pad image so that both images are aligned
        pad_value = abs(image_added_border_left.shape[0] - image_added_border_right.shape[0])
        if curve_down:
            pad_value_top = pad_value
            pad_value_bottom = 0
        else:
            pad_value_top = 0
            pad_value_bottom = pad_value
        if image_added_border_left.shape[0] > image_added_border_right.shape[0]:
            image_added_border_right = np.pad(
                image_added_border_right,
                pad_width=((pad_value_top, pad_value_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if pad_value_top:
                image_added_border_right[:pad_value_top, :] = self.backdrop_color
            else:
                image_added_border_right[-pad_value_bottom:, :] = self.backdrop_color
        elif image_added_border_right.shape[0] > image_added_border_left.shape[0]:
            image_added_border_left = np.pad(
                image_added_border_left,
                pad_width=((pad_value_top, pad_value_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if pad_value_top:
                image_added_border_left[:pad_value_top, :] = self.backdrop_color
            else:
                image_added_border_left[-pad_value_bottom:, :] = self.backdrop_color

        # apply curvy effect
        image_right = self.curve_page(image_added_border_right, curve_value_right)
        if not curve_down:
            image_right = np.flipud(image_right)
        image_left = np.fliplr(self.curve_page(np.fliplr(image_added_border_left), curve_value_left))
        if not curve_down:
            image_left = np.flipud(image_left)

        # further perspective transform
        image_right = self.curve_transform(image_right, self.curve_ratio_right, image_direction="right")
        image_left = self.curve_transform(image_left, self.curve_ratio_left, image_direction="left")

        # pad image so that both images are aligned on the top portion
        if not curve_down:
            pad_value = abs(image_left.shape[0] - image_right.shape[0])
            pad_value_top = pad_value
            if image_left.shape[0] > image_right.shape[0]:
                image_right = np.pad(
                    image_right,
                    pad_width=((pad_value_top, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image_right[:pad_value_top, :] = self.backdrop_color
            elif image_right.shape[0] > image_left.shape[0]:
                image_left = np.pad(
                    image_left,
                    pad_width=((pad_value_top, pad_value_bottom), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image_left[:pad_value_top, :] = self.backdrop_color

        # further bend left image by using perspective transform
        ysize, xsize = image_left.shape[:2]

        # generate range of mirror and crop image based on mirror size
        mirror_range = np.random.uniform(self.mirror_range[0], self.mirror_range[1])
        image_left = image_left[:, image_left.shape[1] - int(image_left.shape[1] * mirror_range) :]

        # get new y and x size of left image
        ysize, xsize = image_left.shape[:2]
        cysize, cxsize = image_right.shape[:2]

        # create new image with original size + mirror size
        image_output = np.full(
            (max(ysize, cysize), xsize + cxsize, image.shape[2]),
            fill_value=self.backdrop_color,
            dtype="uint8",
        )

        # merged left image and right image
        image_output[:ysize, :xsize] = image_left
        image_output[:cysize, xsize:] = image_right

        # add shadow effect (temporary disabled, will be updated later)
        """
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


        image_darken = image_output.astype("float") * 0.7
        offset = random.randint(int(image.shape[0] / 40), int(image.shape[0] / 30))
        image_darken[image_mask > 0] = image_output[image_mask > 0]
        if curve_down:
            image_darken[:-offset, :] = image_darken[offset:, :]
        else:
            image_darken[offset:, :] = image_darken[:-offset, :]
        image_darken = cv2.GaussianBlur(image_darken.astype("uint8"), (151, 151), cv2.BORDER_DEFAULT)
        image_output[image_mask > 0] = image_darken[image_mask > 0]
        """

        return image_output

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # load left side of image from cache
            if self.use_cache_images:
                image_left = load_image_from_cache(random_image=1)
            else:
                image_left = None

            if image_left is not None:

                # resize foreground
                image_left = cv2.resize(
                    image_left,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                # consistent color channel
                if len(image_left) < 3:
                    image_left = cv2.cvtColor(image_left, cv2.COLOR_GRAY2BGR)
            else:
                image_left = image.copy()

            image_output = self.curve_processing(image, image_left)

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

            return image_output

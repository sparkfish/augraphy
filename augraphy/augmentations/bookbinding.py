import random

import cv2
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import four_point_transform
from augraphy.augmentations.lib import load_image_from_cache
from augraphy.augmentations.lib import update_mask_labels
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
    :param enable_shadow: Flag to enable shadow on top of the book binding effect.
    :type enable_shadow: int, optional
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
        enable_shadow=1,
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
        self.enable_shadow = enable_shadow
        self.use_cache_images = use_cache_images
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        return f"BookBinding(shadow_radius_range={self.shadow_radius_range}, curve_range_right={self.curve_range_right}, curve_range_left={self.curve_range_left}, curve_ratio_right={self.curve_ratio_right}, curve_ratio_left={self.curve_ratio_left}, mirror_range={self.mirror_range}, binding_align={self.binding_align}, binding_pages={self.binding_pages}, curling_direction={self.curling_direction}, backdrop_color={self.backdrop_color}, enable_shadow={self.enable_shadow}, use_cache_images={self.use_cache_images}, numba_jit={self.numba_jit}, p={self.p})"

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
        img_d = img_dist + (radius * (1 - np.cos(np.deg2rad(angle))))
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

    def curve_page(self, img, curve_value, backdrop_color, mask=None, keypoints=None, bounding_boxes=None):
        """Generate curvy effect in the input image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param curve_value: Pixel number of the page text should be curved.
        :type curve_value: int
        :param backdrop_color: The color of the filled background.
        :type backdrop_color: tuple
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8), optional
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary, optional
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list, optional
        """
        rows = img.shape[0]
        cols = img.shape[1]

        # curve value not more than image height
        curve_value = min(rows, curve_value)

        if len(img.shape) > 2:
            channels = img.shape[2]
            img_output = np.full(
                (rows + curve_value, cols, channels),
                fill_value=backdrop_color,
                dtype=img.dtype,
            )

        else:
            img_output = np.full((rows + curve_value, cols), fill_value=np.mean(self.backdrop_color), dtype=img.dtype)

        curve_factor = (cols * 2) + (cols * 2 * curve_value / rows)
        self.curve_page_processing(img, img_output, curve_value, rows, cols, curve_factor)

        # apply curve to mask
        if mask is not None:
            mask_output = np.full((rows + curve_value, cols), fill_value=0, dtype=mask.dtype)
            self.curve_page_processing(mask, mask_output, curve_value, rows, cols, curve_factor)
            mask = mask_output

        # apply curve processing to keypoints
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    offset_y = int(curve_value * np.sin(2 * 3.14 * xpoint / curve_factor))
                    points[i] = [xpoint, ypoint + offset_y]

        # apply curve processing to bounding boxes
        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                height = yepoint - yspoint
                offset_y = int(curve_value * np.sin(2 * 3.14 * xspoint / curve_factor))
                bounding_boxes[i] = [xspoint, yspoint + offset_y, xepoint, yspoint + height + offset_y]

        if mask is not None or keypoints is not None or bounding_boxes is not None:
            return [img_output, mask, keypoints, bounding_boxes]
        else:
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
                offset_y = int(curve_value * np.sin(2 * 3.14 * x / curve_factor))
                offseted_y = y + offset_y
                img_output[offseted_y, x] = img[y, x]

        return img_output

    @staticmethod
    @jit(nopython=True, cache=True)
    def check_backdrop_color(image_output, image_mask, backdrop_color):
        """Function to compute mask of background

        :param image_output: The image with BookBinding effect.
        :type image_output: numpy.array (numpy.uint8)
        :param image_mask: The mask of background.
        :type image_mask: numpy.array (numpy.uint8)
        :param backdrop_color: The backdrop color in BGR format.
        :type backdrop_color: tuple
        """

        for x in range(image_mask.shape[1]):
            # check top down
            for y in range(image_mask.shape[0]):
                image_mask[y, x] = backdrop_color
                color_eval = (
                    image_output[y, x][0] == backdrop_color[0]
                    and image_output[y, x][1] == backdrop_color[1]
                    and image_output[y, x][2] == backdrop_color[2]
                )
                if not color_eval:
                    break
            # # check bottom up : reverse loop
            for y in range(image_mask.shape[0] - 1, -1, -1):
                image_mask[y, x] = backdrop_color
                color_eval = (
                    image_output[y, x][0] == backdrop_color[0]
                    and image_output[y, x][1] == backdrop_color[1]
                    and image_output[y, x][2] == backdrop_color[2]
                )
                if not color_eval:
                    break

        for y in range(image_mask.shape[0]):
            # check left right
            for x in range(image_mask.shape[1]):
                image_mask[y, x] = backdrop_color
                color_eval = (
                    image_output[y, x][0] == backdrop_color[0]
                    and image_output[y, x][1] == backdrop_color[1]
                    and image_output[y, x][2] == backdrop_color[2]
                )
                if not color_eval:
                    break
            # # check right left : reverse loop
            for x in range(image_mask.shape[1] - 1, -1, -1):
                image_mask[y, x] = backdrop_color
                color_eval = (
                    image_output[y, x][0] == backdrop_color[0]
                    and image_output[y, x][1] == backdrop_color[1]
                    and image_output[y, x][2] == backdrop_color[2]
                )
                if not color_eval:
                    break

    def curve_transform(
        self,
        image,
        curve_ratio_range,
        image_direction,
        mask=None,
        keypoints=None,
        bounding_boxes=None,
    ):
        """Bend image further in horizontal direction using perspective transform.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param curve_ratio: The ratio of transformation in horizontal direction..
        :type curve_ratio: float
        :param image_direction: Flag to identify left or right side of image.
        :type image_direction: string
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8), optional
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary, optional
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list, optional
        """

        curve_ratio = random.uniform(curve_ratio_range[0], curve_ratio_range[1])

        ysize, xsize = image.shape[:2]

        # start and end points for left and right side of images
        if image_direction == "left":
            new_curve_ratio = curve_ratio
            source_points = np.float32([[0, 0], [xsize, 0], [xsize, ysize], [0, ysize]])
            target_points = np.float32(
                [[int(xsize * new_curve_ratio), 0], [xsize, 0], [xsize, ysize], [int(xsize * new_curve_ratio), ysize]],
            )
        else:
            new_curve_ratio = 1 - curve_ratio
            source_points = np.float32([[0, 0], [xsize, 0], [xsize, ysize], [0, ysize]])
            target_points = np.float32(
                [[0, 0], [int(xsize * new_curve_ratio), 0], [int(xsize * new_curve_ratio), ysize], [0, ysize]],
            )

        # perspective transform to further bend image in x direction only
        image_transformed = four_point_transform(image, source_points, target_points, xsize, ysize)

        # transform mask
        if mask is not None:
            mask_labels = np.unique(mask).tolist() + [0]
            mask = four_point_transform(mask, source_points, target_points, xsize, ysize)
            update_mask_labels(mask, mask_labels)

        # crop the blank area after the transform
        if image_direction == "left":
            image_transformed = image_transformed[:, int(xsize * curve_ratio) :]
            if mask is not None:
                mask = mask[:, int(xsize * curve_ratio) :]
        else:
            image_transformed = image_transformed[:, : int(xsize * (1 - curve_ratio))]
            if mask is not None:
                mask = mask[:, : int(xsize * (1 - curve_ratio))]

        # transform keypoints
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [int(xpoint * new_curve_ratio), ypoint]

        # transform bounding boxes
        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    int(xspoint * new_curve_ratio),
                    yspoint,
                    int(xepoint * new_curve_ratio),
                    yepoint,
                ]

        if mask is not None or keypoints is not None or bounding_boxes is not None:
            return [image_transformed.astype("uint8"), mask, keypoints, bounding_boxes]
        else:
            return image_transformed.astype("uint8")

    def curve_processing(self, image, image_left, mask=None, keypoints=None, bounding_boxes=None):
        """Core function for curvy effect processing.

        :param image: The right image of the book binding effect.
        :type image: numpy.array (numpy.uint8)
        :param image_left: The left image of the book binding effect.
        :type image_left: numpy.array (numpy.uint8)
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8), optional
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary, optional
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list, optional
        """

        # min value is 1, to differentiate between background
        backdrop_color = (
            max(1, self.backdrop_color[0]),
            max(1, self.backdrop_color[1]),
            max(1, self.backdrop_color[2]),
        )

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
            page_border_background_color=backdrop_color,
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
            iysize, ixsize = image.shape[:2]
            # flip image
            image = np.flipud(image)
            # flip mask
            if mask is not None:
                mask = np.flipud(mask)
            # flip keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xpoint, iysize - 1 - ypoint]
            # flip bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    bounding_boxes[i] = [
                        xspoint,
                        iysize - 1 - yspoint,
                        xepoint,
                        iysize - 1 - yepoint,
                    ]

        image_shadow = self.add_book_shadow(image, radius)
        image_added_border_right = page_border(
            image=image_shadow,
            mask=mask,
            keypoints=keypoints,
            bounding_boxes=bounding_boxes,
        )
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            image_added_border_right, mask, keypoints, bounding_boxes = image_added_border_right

        # left side of image
        # create borders
        page_border = PageBorder(
            page_border_width_height=width_height,
            page_border_color=(0, 0, 0),
            page_border_background_color=backdrop_color,
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
            # pad image
            image_added_border_right = np.pad(
                image_added_border_right,
                pad_width=((pad_value_top, pad_value_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if pad_value_top:
                image_added_border_right[:pad_value_top, :] = backdrop_color
            else:
                image_added_border_right[-pad_value_bottom:, :] = backdrop_color
            # pad mask
            if mask is not None:
                mask = np.pad(
                    mask,
                    pad_width=((pad_value_top, pad_value_bottom), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # pad keypoints
            if keypoints is not None and pad_value_top != 0:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xpoint, ypoint + pad_value_top]

            # pad bounding boxes
            if bounding_boxes is not None and pad_value_top != 0:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    bounding_boxes[i] = [
                        xspoint,
                        yspoint + pad_value_top,
                        xepoint,
                        yepoint + pad_value_top,
                    ]

        elif image_added_border_right.shape[0] > image_added_border_left.shape[0]:
            image_added_border_left = np.pad(
                image_added_border_left,
                pad_width=((pad_value_top, pad_value_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if pad_value_top:
                image_added_border_left[:pad_value_top, :] = backdrop_color
            else:
                image_added_border_left[-pad_value_bottom:, :] = backdrop_color

        # apply curvy effect
        # right
        image_right = self.curve_page(
            image_added_border_right,
            curve_value_right,
            backdrop_color,
            mask,
            keypoints,
            bounding_boxes,
        )
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            image_right, mask, keypoints, bounding_boxes = image_right
        if not curve_down:
            iysize, ixsize = image_right.shape[:2]
            # flip image
            image_right = np.flipud(image_right)
            # flip mask
            if mask is not None:
                mask = np.flipud(mask)
            # flip keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xpoint, iysize - 1 - ypoint]
            # flip bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    bounding_boxes[i] = [
                        xspoint,
                        iysize - 1 - yspoint,
                        xepoint,
                        iysize - 1 - yepoint,
                    ]
        # left
        image_left = np.fliplr(self.curve_page(np.fliplr(image_added_border_left), curve_value_left, backdrop_color))
        if not curve_down:
            image_left = np.flipud(image_left)

        # further perspective transform
        image_right = self.curve_transform(
            image=image_right,
            curve_ratio_range=self.curve_ratio_right,
            image_direction="right",
            mask=mask,
            keypoints=keypoints,
            bounding_boxes=bounding_boxes,
        )
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            image_right, mask, keypoints, bounding_boxes = image_right
        image_left = self.curve_transform(
            image=image_left,
            curve_ratio_range=self.curve_ratio_left,
            image_direction="left",
        )

        # pad image so that both images are aligned on the top portion
        if not curve_down:
            pad_value = abs(image_left.shape[0] - image_right.shape[0])
            pad_value_top = pad_value
            if image_left.shape[0] > image_right.shape[0]:
                # pad image
                image_right = np.pad(
                    image_right,
                    pad_width=((pad_value_top, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image_right[:pad_value_top, :] = backdrop_color
                # pad mask
                if mask is not None:
                    mask = np.pad(
                        mask,
                        pad_width=((pad_value_top, 0), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                # pad keypoints
                if keypoints is not None and pad_value_top != 0:
                    for name, points in keypoints.items():
                        for i, (xpoint, ypoint) in enumerate(points):
                            points[i] = [xpoint, ypoint + pad_value_top]

                # pad bounding boxes
                if bounding_boxes is not None and pad_value_top != 0:
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        bounding_boxes[i] = [
                            xspoint,
                            yspoint + pad_value_top,
                            xepoint,
                            yepoint + pad_value_top,
                        ]

            elif image_right.shape[0] > image_left.shape[0]:
                image_left = np.pad(
                    image_left,
                    pad_width=((pad_value_top, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image_left[:pad_value_top, :] = backdrop_color

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
            fill_value=backdrop_color,
            dtype="uint8",
        )

        # merged left image and right image
        image_output[:ysize, :xsize] = image_left
        image_output[:cysize, xsize:] = image_right

        # add image left to mask, keypoints and bounding boxes by ading offset or padding
        # pad mask with image left xsize
        if mask is not None:
            pad_y = 0
            if cysize < ysize:
                pad_y = ysize - cysize

            mask = np.pad(
                mask,
                pad_width=((0, pad_y), (xsize, 0)),
                mode="constant",
                constant_values=0,
            )

        # add image left xsize as offset to keypoints
        if keypoints is not None and pad_value_top != 0:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [xpoint + xsize, ypoint]

        # add image left xsize as offset to bounding boxes
        if bounding_boxes is not None and pad_value_top != 0:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    xspoint + xsize,
                    yspoint,
                    xepoint + xsize,
                    yepoint,
                ]

        # add shadow effect
        if self.enable_shadow:
            # get mask for shadow effect
            image_mask = np.zeros_like(image_output, dtype="uint8")

            self.check_backdrop_color(image_output, image_mask, backdrop_color)

            mysize, mxsize = image_mask.shape[:2]
            # extended value
            ext = 100
            emysize, emxsize = (
                mysize + (ext * 2),
                mxsize + (ext * 2),
            )
            image_mask_extend = np.full((emysize, emxsize, 3), fill_value=backdrop_color, dtype="uint8")
            image_mask_extend[ext:-ext, ext:-ext] = image_mask

            image_shadow = cv2.resize(
                image_mask_extend,
                (max(50, int(emxsize / 10)), max(50, int(emysize / 10))),
                interpolation=cv2.INTER_AREA,
            )
            image_shadow = cv2.GaussianBlur(image_shadow, (255, 255), cv2.BORDER_DEFAULT)
            image_shadow = cv2.resize(image_shadow, (emxsize, emysize), interpolation=cv2.INTER_LINEAR)
            image_shadow = image_shadow[ext:-ext, ext:-ext]
            image_output[image_mask > 0] = image_shadow[image_mask > 0]

            # replace with input backdrop_color
            for i in range(3):
                if backdrop_color[i] != self.backdrop_color[i]:
                    indices = np.logical_and(image_output[:, :, i] == backdrop_color[i], image_mask[:, :, i] > 0)
                    image_output[:, :, i][indices] = self.backdrop_color[i]

        return image_output, mask, keypoints, bounding_boxes

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
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

            image_output, mask, keypoints, bounding_boxes = self.curve_processing(
                image=image,
                image_left=image_left,
                mask=mask,
                keypoints=keypoints,
                bounding_boxes=bounding_boxes,
            )

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                ysize, xsize = image_output.shape[:2]
                if ysize != image_alpha.shape[0] or xsize != image_alpha.shape[1]:
                    image_alpha = np.full((ysize, xsize), fill_value=255, dtype="uint8")
                image_output = np.dstack((image_output, image_alpha))

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_output] + outputs_extra
            else:
                return image_output

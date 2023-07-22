import random

import cv2
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import load_image_from_cache
from augraphy.augmentations.lib import rotate_image_PIL
from augraphy.augmentations.lib import warp_fold
from augraphy.base.augmentation import Augmentation


class PageBorder(Augmentation):
    """Add page border effect by stacking images multiple times.

    :param page_border_width_height: Pair of values determining the width and height of the page border effect.
            If width > 0, the effect applies in right side of the page, while if width <0, the effect applies in the left side of the page.
            If height > 0, the effect applies in bottom side of the page, while if height <0, the effect applies in the top side of the page.
            If the value is within the range of -1.0 to 1.0 and the value is float,
            border width will be scaled by image width, while border height will be sccaled by image height.
            width (int) = image width  * width (float and -1.0 - 1.0);
            height (int) = image height * height (float and -1.0 - 1.0);
            Default value is "random".
    :type page_border_width_height: tuple or string , optional
    :param page_border_color: The color (BGR) of border effect.
    :type page_border_color: tuple, optional
    :param page_border_background_color: The color (BGR) of border background.
    :type page_border_background_color: tuple, optional
    :param page_border_use_cache_images: Flag to enable the usage of cache images in creating page border effect.
    :type page_border_use_cache_images: int, optional
    :param page_border_trim_sides： Tuple of 4 (left, top, right, bottom) determining which sides of the image to be trimmed.
            This is valid only if same_page_border is false.
    :type page_border_trim_sides: int, optional
    :param page_numbers: An integer determining the number of pages in the border.
    :type page_numbers: int, optional
    :param page_rotation_angle_in_order: Flag to enable an ordered or random angle rotation.
    :type page_rotation_angle_in_order: int , optional
    :param page_rotation_angle_range: Pair of ints determining the angle of rotation in the border effect.
    :type page_rotation_angle_range: tuple , optional
    :param curve_frequency: Pair of ints determining number of curvy section in the generated page borders.
    :type curve_frequency: tuple, optional
    :param curve_height: Pair of ints determining height of curvy section in the generated page borders.
    :type curve_height: tuple, optional
    :param curve_length_one_side: Pair of ints determining one side length of generated curvy effect.
    :type curve_length_one_side: tuple, optional
    :param same_page_border: Flag to decide whether the added borders will be within the input image or not.
    :type same_page_border: int, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        page_border_width_height="random",
        page_border_color=(0, 0, 0),
        page_border_background_color=(0, 0, 0),
        page_border_use_cache_images=0,
        page_border_trim_sides=(0, 0, 0, 0),
        page_numbers="random",
        page_rotate_angle_in_order=1,
        page_rotation_angle_range=(-3, 3),
        curve_frequency=(0, 1),
        curve_height=(2, 4),
        curve_length_one_side=(50, 100),
        same_page_border=1,
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p, numba_jit=numba_jit)

        self.page_border_width_height = page_border_width_height
        self.page_border_color = page_border_color
        self.page_border_background_color = page_border_background_color
        self.page_border_use_cache_images = page_border_use_cache_images
        self.page_border_trim_sides = page_border_trim_sides
        self.page_numbers = page_numbers
        self.page_rotate_angle_in_order = page_rotate_angle_in_order
        self.page_rotation_angle_range = page_rotation_angle_range
        self.curve_frequency = curve_frequency
        self.curve_height = curve_height
        self.curve_length_one_side = curve_length_one_side
        self.same_page_border = same_page_border
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        return f"PageBorder(page_border_width_height={self.page_border_width_height}, page_border_color={self.page_border_color}, page_border_background_color={self.page_border_background_color}, page_border_use_cache_images={self.page_border_use_cache_images},  page_border_trim_sides={self.page_border_trim_sides}, page_numbers={self.page_numbers}, page_rotate_angle_in_order={self.page_rotate_angle_in_order}, page_rotation_angle_range={self.page_rotation_angle_range}, curve_frequency={self.curve_frequency}, curve_height={self.curve_height}, curve_length_one_side={self.curve_length_one_side}, same_page_border={self.same_page_border}, numba_jit={self.numba_jit}, p={self.p})"

    def random_folding(self, image):
        """Create random folding effect at the image border.

        :param Image: Image to be folded.
        :type image: numpy.array (numpy.uint8)
        """

        # get image x and y size
        ysize, xsize = image.shape[:2]

        # height of curve, min value is 1
        curve_y_shift = random.randint(
            max(1, self.curve_height[0]),
            max(1, self.curve_height[1]),
        )

        # length of one side curvy part, min value is 5
        curve_width_one_side = random.randint(
            max(5, self.curve_length_one_side[0]),
            max(5, self.curve_length_one_side[1]),
        )

        # prevent random randint second value is smaller than first value
        while xsize - curve_width_one_side - 1 < curve_width_one_side + 1:
            curve_width_one_side = max(1, int(curve_width_one_side / 2))

        # center of curvy part
        curve_x = random.randint(
            curve_width_one_side + 1,
            xsize - curve_width_one_side - 1,
        )

        # filler of folding function
        curve_noise = 0

        # warp image to produce curvy effect
        image_curve_left = warp_fold(
            image,
            ysize,
            curve_noise,
            curve_x,
            curve_width_one_side,
            curve_y_shift,
            side="left",
            backdrop_color=self.page_border_background_color,
        )

        image_curve_right = warp_fold(
            image_curve_left,
            ysize,
            curve_noise,
            curve_x,
            curve_width_one_side,
            curve_y_shift,
            side="right",
            backdrop_color=self.page_border_background_color,
        )

        image_color = np.full_like(image_curve_right, fill_value=self.page_border_color, dtype="uint8")
        image_curve_right[image_curve_right == (0, 0, 0)] = image_color[image_curve_right == (0, 0, 0)]

        return image_curve_right

    def center_overlay(self, image_background, image_foreground, image_mask_background, image_mask_foreground):
        """Overlays foreground image into background image in the center.

        :param image_background: The background image.
        :type image_background: numpy array
        :param image_foreground: The foreground image.
        :type image_foreground: numpy array
        :param image_mask_background: The mask of background image.
        :type image_mask_background: numpy array
        :param image_mask_foreground: The mask of foreground image.
        :type image_mask_foreground: numpy array
        """

        ysize, xsize = image_foreground.shape[:2]
        bysize, bxsize = image_background.shape[:2]

        # y
        ycenter = int(ysize / 2)
        bycenter = int(bysize / 2)
        dy_top = abs(ycenter - bycenter)
        dy_bottom = abs(abs(ysize - ycenter) - abs(bysize - bycenter))
        # x
        xcenter = int(xsize / 2)
        bxcenter = int(bxsize / 2)
        dy_left = abs(xcenter - bxcenter)
        dy_right = abs(abs(xsize - xcenter) - abs(bxsize - bxcenter))

        # condition where foreground image is larger
        if ysize > bysize:
            # (top, bottom), (left, right)
            image_background = np.pad(
                image_background,
                pad_width=((dy_top, dy_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            image_mask_background = np.pad(
                image_mask_background,
                pad_width=((dy_top, dy_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if xsize > bxsize:
            # (top, bottom), (left, right)
            image_background = np.pad(
                image_background,
                pad_width=((0, 0), (dy_left, dy_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            image_mask_background = np.pad(
                image_mask_background,
                pad_width=((0, 0), (dy_left, dy_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        indices = image_mask_foreground > 0

        # overlay image
        if bysize > ysize and bxsize > xsize:
            image_background[dy_top:-dy_bottom, dy_left:-dy_right][indices] = image_foreground[indices]
            image_mask_background[dy_top:-dy_bottom, dy_left:-dy_right] += image_mask_foreground
        elif bysize > ysize:
            image_background[dy_top:-dy_bottom, :][indices] = image_foreground[indices]
            image_mask_background[dy_top:-dy_bottom, :] += image_mask_foreground
        elif bxsize > xsize:
            image_background[:, dy_left:-dy_right][indices] = image_foreground[indices]
            image_mask_background[:, dy_left:-dy_right] += image_mask_foreground
        else:
            image_background[indices] = image_foreground[indices]
            image_mask_background += image_mask_foreground

        return image_background, image_mask_background

    def create_page_borders(
        self,
        image,
        page_border_width,
        page_border_height,
    ):
        """Create page borders effect and apply it into input image.

        :param image: The input image.
        :type image: numpy array
        :param border_width: Horizontal direction and width of the borders.
        :type border_width: int
        :param border_height: Vertical direction and height of borders.
        :type border_height: int
        """

        border_width = abs(page_border_width)
        border_height = abs(page_border_height)

        # temporary, in case we need random input
        page_border_background_color = self.page_border_background_color

        # generate page number
        if self.page_numbers == "random":
            min_length = min(max(1, abs(page_border_width)), max(1, abs(page_border_height)))
            min_page = max(1, int(min_length / 10))
            max_page = max(1, int(min_length / 8))
            page_numbers = random.randint(min_page, max_page)
        else:
            page_numbers = self.page_numbers
        # +1 to have an extra top layer page
        page_numbers += 1

        ysize, xsize = image.shape[:2]

        # load border images
        border_images = []
        if self.page_border_use_cache_images:
            for i in range(page_numbers - 1):
                image_cache = load_image_from_cache(random_image=1)
                if image_cache is None:
                    image_cache = image.copy()
                else:
                    image_cache = cv2.resize(
                        image_cache,
                        (xsize, ysize),
                        interpolation=cv2.INTER_AREA,
                    )
                border_images.append(image_cache)
            # last image on the surface must be the input image
            border_images.append(image.copy())
        else:
            for i in range(page_numbers):
                border_images.append(image.copy())

        yxsize = []
        for i, border_image in enumerate(border_images):
            # default, extend top left
            if page_border_width < 0 and page_border_height < 0:
                page_border_trim_sides = self.page_border_trim_sides
            # extend bottom left
            elif page_border_width < 0 and page_border_height > 0:
                # rotate counter clockwise 3 times to align bottomleft to topleft （topleft is reference)
                border_image = np.rot90(border_image, 3)
                page_border_trim_sides = (
                    self.page_border_trim_sides[3],
                    self.page_border_trim_sides[0],
                    self.page_border_trim_sides[1],
                    self.page_border_trim_sides[2],
                )
            # extend bottom right
            elif page_border_width > 0 and page_border_height > 0:
                # rotate counter clockwise twice to align bottomright to topleft （topleft is reference)
                border_image = np.rot90(border_image, 2)
                page_border_trim_sides = (
                    self.page_border_trim_sides[2],
                    self.page_border_trim_sides[3],
                    self.page_border_trim_sides[0],
                    self.page_border_trim_sides[1],
                )
            # extend top right
            elif page_border_width > 0 and page_border_height < 0:
                # rotate counter clockwise once to align topright to topleft （topleft is reference)
                border_image = np.rot90(border_image, 1)
                page_border_trim_sides = (
                    self.page_border_trim_sides[1],
                    self.page_border_trim_sides[2],
                    self.page_border_trim_sides[3],
                    self.page_border_trim_sides[0],
                )
            # extend top, default
            elif page_border_width == 0 and page_border_height < 0:
                page_border_trim_sides = self.page_border_trim_sides
            # bottom only
            elif page_border_width == 0 and page_border_height > 0:
                # rotate counter clockwise twice to align bottom to top （top is reference)
                border_image = np.rot90(border_image, 2)
                page_border_trim_sides = (
                    self.page_border_trim_sides[2],
                    self.page_border_trim_sides[3],
                    self.page_border_trim_sides[0],
                    self.page_border_trim_sides[1],
                )
            # extend left, default
            elif page_border_width < 0 and page_border_height == 0:
                page_border_trim_sides = self.page_border_trim_sides
            # right only
            elif page_border_width > 0 and page_border_height == 0:
                # rotate counter clockwise twice to align right to left （left is reference)
                border_image = np.rot90(border_image, 2)
                page_border_trim_sides = (
                    self.page_border_trim_sides[2],
                    self.page_border_trim_sides[3],
                    self.page_border_trim_sides[0],
                    self.page_border_trim_sides[1],
                )

            # get size before the pruning for same page border
            if not yxsize:
                rysize, rxsize = border_image.shape[:2]
                yxsize.extend([rysize, rxsize])

            # for same page border, page border grows internally
            if self.same_page_border:
                if border_width == 0:
                    border_image = border_image[border_height:, :]
                elif border_height == 0:
                    border_image = border_image[:, border_width:]
                else:
                    border_image = border_image[border_height:, border_width:]

            border_images[i] = border_image

        # create background image
        if self.same_page_border:
            border_image_merged = np.full_like(border_images[0], fill_value=page_border_background_color, dtype="uint8")
        else:
            border_image_merged = np.full(
                (border_images[0].shape[0] + border_height, border_images[1].shape[1] + border_width, 3),
                fill_value=page_border_background_color,
                dtype="uint8",
            )

        # interpolate shifting x and y
        if border_width == 0:
            shifted_value_xs = [0 for _ in range(page_numbers)]
            shifted_value_ys = np.linspace(border_height, 0, page_numbers)
        elif border_height == 0:
            shifted_value_xs = np.linspace(border_width, 0, page_numbers)
            shifted_value_ys = [0 for _ in range(page_numbers)]
        else:
            shifted_value_xs = np.linspace(border_width, 0, page_numbers)
            shifted_value_ys = np.linspace(border_height, 0, page_numbers)

        # total number of page shifting process
        total_shifts = page_numbers

        # angle
        rotated_angle = random.randint(self.page_rotation_angle_range[0], self.page_rotation_angle_range[1])
        rotated_angle_step = rotated_angle / (total_shifts)

        image_mask_background = np.full_like(border_image_merged, fill_value=0, dtype="uint8")

        for i in reversed(range(total_shifts)):

            shifted_value_y = int(shifted_value_ys[i])
            shifted_value_x = int(shifted_value_xs[i])

            # create a copy of image
            border_image_fold = border_images[total_shifts - 1 - i].copy()

            # draw borders line on possible folding edges before folding process
            if page_border_height != 0:
                border_image_fold[0, :] = self.page_border_color
            if page_border_width != 0:
                border_image_fold[:, 0] = self.page_border_color

            # apply curve effect to borders (last image shouldn't apply folding)
            if i != 0:
                curve_frequency = random.randint(self.curve_frequency[0], self.curve_frequency[1])
                for _ in range(curve_frequency):
                    border_image_fold = self.random_folding(border_image_fold)
                    border_image_fold = np.rot90(self.random_folding(np.rot90(border_image_fold, 3)))

            border_image_single = border_image_fold.copy()
            # draw the rest of borders line
            border_image_single[-1, :] = self.page_border_color
            border_image_single[0, :] = self.page_border_color
            border_image_single[:, 0] = self.page_border_color
            border_image_single[:, -1] = self.page_border_color
            image_mask_rotate_single = np.full_like(border_image_single, fill_value=1, dtype="uint8")

            # pad image based on shift values
            extend_bottom = border_height - shifted_value_y
            extend_right = border_width - shifted_value_x
            border_image_single = np.pad(
                border_image_single,
                pad_width=((shifted_value_y, extend_bottom), (shifted_value_x, extend_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            image_mask_rotate_single = np.pad(
                image_mask_rotate_single,
                pad_width=((shifted_value_y, extend_bottom), (shifted_value_x, extend_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            # rotate page
            if self.page_rotate_angle_in_order:
                # rotation
                rotated_angle -= rotated_angle_step
            else:
                # no rotation on final image
                if i == 0:
                    rotated_angle = 0
                else:
                    rotated_angle = random.randint(self.page_rotation_angle_range[0], self.page_rotation_angle_range[1])
            if rotated_angle != 0:
                image_mask_rotate_single = rotate_image_PIL(image_mask_rotate_single, rotated_angle, expand=1)
                border_image_single = rotate_image_PIL(border_image_single, rotated_angle, expand=1)

            border_image_merged, image_mask_background = self.center_overlay(
                border_image_merged,
                border_image_single,
                image_mask_background,
                image_mask_rotate_single,
            )

        # update background value based on mask
        image_mask_background[image_mask_background > 0] = 255
        image_mask_background_gray = cv2.cvtColor(image_mask_background, cv2.COLOR_BGR2GRAY)
        indices = image_mask_background_gray == 0
        border_image_merged[indices] = page_border_background_color

        border_image_merged = cv2.GaussianBlur(border_image_merged, (3, 3), 0)

        # merge each pages into a main page
        if self.same_page_border or sum(page_border_trim_sides) > 0:

            ysize, xsize = yxsize
            bysize, bxsize = border_image_merged.shape[:2]

            if bysize > ysize or bxsize > xsize:

                # y
                ycenter = int(ysize / 2)
                bycenter = int(bysize / 2)
                dy_top = abs(ycenter - bycenter)
                dy_bottom = abs(abs(ysize - ycenter) - abs(bysize - bycenter))
                # x
                xcenter = int(xsize / 2)
                bxcenter = int(bxsize / 2)
                dx_left = abs(xcenter - bxcenter)
                dx_right = abs(abs(xsize - xcenter) - abs(bxsize - bxcenter))

                if not self.same_page_border and sum(page_border_trim_sides) > 0:
                    half_border_width = int(np.ceil(border_width / 2))
                    half_border_height = int(np.ceil(border_height / 2))
                    dy_top += half_border_height
                    dy_bottom -= half_border_height
                    dx_left += half_border_width
                    dx_right -= half_border_width

                if bysize > ysize and bxsize > xsize:
                    if self.same_page_border:
                        border_image_merged = border_image_merged[dy_top:-dy_bottom, dx_left:-dx_right]
                    else:
                        if page_border_trim_sides[0]:
                            start_x = dx_left
                        else:
                            start_x = 0
                        if page_border_trim_sides[1]:
                            start_y = dy_top
                        else:
                            start_y = 0
                        if page_border_trim_sides[2]:
                            end_x = -dx_right
                        else:
                            end_x = bxsize
                        if page_border_trim_sides[3]:
                            end_y = -dy_bottom
                        else:
                            end_y = bysize
                        border_image_merged = border_image_merged[start_y:end_y, start_x:end_x]

                elif bysize > ysize:
                    if self.same_page_border:
                        border_image_merged = border_image_merged[dy_top:-dy_bottom, :]
                    else:
                        if page_border_trim_sides[1]:
                            start_y = dy_top
                        else:
                            start_y = 0
                        if page_border_trim_sides[3]:
                            end_y = -dy_bottom
                        else:
                            end_y = bysize
                        border_image_merged = border_image_merged[start_y:end_y, :]

                elif bxsize > xsize:
                    if self.same_page_border:
                        border_image_merged = border_image_merged[:, dx_left:-dx_right]
                    else:
                        if page_border_trim_sides[0]:
                            start_x = dx_left
                        else:
                            start_x = 0
                        if page_border_trim_sides[2]:
                            end_x = -dx_right
                        else:
                            end_x = bxsize
                        border_image_merged = border_image_merged[:, start_x:end_x]

        # rotate back to original position
        # default, extend top left
        if page_border_width < 0 and page_border_height < 0:
            pass
        # bottom left
        elif page_border_width < 0 and page_border_height > 0:
            # rotate counter clockwise once from topleft （topleft is reference) back to bottomleft
            border_image_merged = np.rot90(border_image_merged, 1)
        # bottom right
        elif page_border_width > 0 and page_border_height > 0:
            # rotate counter clockwise twice from topleft （topleft is reference) back to bottomleft
            border_image_merged = np.rot90(border_image_merged, 2)
        # top right
        elif page_border_width > 0 and page_border_height < 0:
            # rotate counter clockwise 3 times from topleft （topleft is reference) back to topright
            border_image_merged = np.rot90(border_image_merged, 3)
        # top
        elif page_border_width == 0 and page_border_height < 0:
            pass
        # bottom
        elif page_border_width == 0 and page_border_height > 0:
            # rotate counter clockwise twice from top (top is reference) back to bottom
            border_image_merged = np.rot90(border_image_merged, 2)
        # left
        elif page_border_width < 0 and page_border_height == 0:
            pass
        # right
        elif page_border_width > 0 and page_border_height == 0:
            # rotate counter clockwise 2 times  from left (left is reference) back to right
            border_image_merged = np.rot90(border_image_merged, 2)

        return border_image_merged

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            height, width = image.shape[:2]

            # generate width and height
            if self.page_border_width_height == "random":
                border_width = int(random.uniform(0.01, 0.05) * width) * random.choice([1, -1])
                border_height = int(random.uniform(0.01, 0.05) * height) * random.choice([1, -1])
            else:
                # check if value is flaot and scale them with image width or height
                if (
                    self.page_border_width_height[0] <= 1
                    and self.page_border_width_height[0] >= -1
                    and isinstance(self.page_border_width_height[0], float)
                ):
                    border_width = int(self.page_border_width_height[0] * width)
                else:
                    border_width = self.page_border_width_height[0]

                if (
                    self.page_border_width_height[1] <= 1
                    and self.page_border_width_height[1] >= -1
                    and isinstance(self.page_border_width_height[1], float)
                ):
                    border_height = int(self.page_border_width_height[1] * height)
                else:
                    border_height = self.page_border_width_height[1]

            image_output = self.create_page_borders(
                image.copy(),
                border_width,
                border_height,
            )

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

            return image_output

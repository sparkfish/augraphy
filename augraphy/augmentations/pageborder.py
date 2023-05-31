import random

import cv2
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import rotate_image
from augraphy.augmentations.lib import warp_fold_left_side
from augraphy.augmentations.lib import warp_fold_right_side
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
    :param page_numbers: An integer determining the number of pages in the border.
    :type page_numbers: int, optional
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
        page_numbers="random",
        page_rotation_angle_range=(-3, 3),
        curve_frequency=(2, 8),
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
        self.page_numbers = page_numbers
        self.page_rotation_angle_range = page_rotation_angle_range
        self.curve_frequency = curve_frequency
        self.curve_height = curve_height
        self.curve_length_one_side = curve_length_one_side
        self.same_page_border = same_page_border
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        return f"PageBorder(page_border_width_height={self.page_border_width_height}, page_border_color={self.page_border_color}, page_border_background_color={self.page_border_background_color}, page_numbers={self.page_numbers}, page_rotation_angle_range={self.page_rotation_angle_range}, curve_frequency={self.curve_frequency}, curve_height={self.curve_height}, curve_length_one_side={self.curve_length_one_side}, same_page_border={self.same_page_border}, numba_jit={self.numba_jit}, p={self.p})"

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

        # center of curvy part
        curve_x = random.randint(
            curve_width_one_side + 1,
            xsize - curve_width_one_side - 1,
        )

        # filler of folding function
        curve_noise = 0

        # warp image to produce curvy effect
        image_curve_left = warp_fold_left_side(
            image,
            ysize,
            curve_noise,
            curve_x,
            curve_width_one_side,
            curve_y_shift,
        )

        image_curve_right = warp_fold_right_side(
            image_curve_left,
            ysize,
            curve_noise,
            curve_x,
            curve_width_one_side,
            curve_y_shift,
        )

        image_color = np.full_like(image_curve_right, fill_value=self.page_border_color, dtype="uint8")
        image_curve_right[image_curve_right == (0, 0, 0)] = image_color[image_curve_right == (0, 0, 0)]

        return image_curve_right

    @staticmethod
    @jit(nopython=True, cache=True)
    def set_invalid_value(border_image_single):
        """Change the filled value from rotation into -1 value.

        :param border_image_single: The input image.
        :type border_image_single: numpy array

        """
        bysize, bxsize = border_image_single.shape[:2]

        # check from top left
        for cx in range(bxsize):
            check = 0
            for cy in range(bysize):
                if np.all(border_image_single[cy, cx] == 0):
                    border_image_single[cy, cx] = -1
                    check = 1
                else:
                    if cy - 1 >= 0:
                        border_image_single[cy - 1, cx] = 1

                    break

            if not check:
                break

        return border_image_single

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

        border_image = image.copy()

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

        # default, extend top left
        if page_border_width < 0 and page_border_height < 0:
            pass
        # extend bottom left
        elif page_border_width < 0 and page_border_height > 0:
            # rotate counter clockwise 3 times to align bottomleft to topleft （topleft is reference)
            border_image = np.rot90(border_image, 3)
        # extend bottom right
        elif page_border_width > 0 and page_border_height > 0:
            # rotate counter clockwise twice to align bottomright to topleft （topleft is reference)
            border_image = np.rot90(border_image, 2)
        # extend top right
        elif page_border_width > 0 and page_border_height < 0:
            # rotate counter clockwise once to align topright to topleft （topleft is reference)
            border_image = np.rot90(border_image, 1)
        # extend top, default
        elif page_border_width == 0 and page_border_height < 0:
            pass
        # bottom only
        elif page_border_width == 0 and page_border_height > 0:
            # rotate counter clockwise twice to align bottom to top （top is reference)
            border_image = np.rot90(border_image, 2)
        # extend left, default
        elif page_border_width < 0 and page_border_height == 0:
            pass
        # right only
        elif page_border_width > 0 and page_border_height == 0:
            # rotate counter clockwise twice to align right to left （left is reference)
            border_image = np.rot90(border_image, 2)

        image_rotated = border_image.copy()
        border_width = abs(page_border_width)
        border_height = abs(page_border_height)

        # min color value is 1, to prevent a same black background on the rotated image
        page_border_background_color = list(self.page_border_background_color)
        for i in range(3):
            page_border_background_color[i] = max(page_border_background_color[i], 1)

        # for same page border, page border grows internally
        if self.same_page_border:
            if border_width == 0:
                border_image = border_image[border_height:, :]
            elif border_height == 0:
                border_image = border_image[:, border_width:]
            else:
                border_image = border_image[border_height:, border_width:]

            border_image_merged = np.full_like(image_rotated, fill_value=page_border_background_color, dtype="uint8")
        else:
            border_image_merged = np.full(
                (image_rotated.shape[0] + border_height, image_rotated.shape[1] + border_width, 3),
                fill_value=page_border_background_color,
                dtype="uint8",
            )

        border_image_single = border_image.copy()

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

        for i in reversed(range(total_shifts)):

            shifted_value_y = int(shifted_value_ys[i])
            shifted_value_x = int(shifted_value_xs[i])

            # create a copy of image
            border_image_fold = border_image.copy()

            # draw borders line on possible folding edges before folding process
            if page_border_height != 0:
                border_image_fold[0, :] = self.page_border_color
            if page_border_width != 0:
                border_image_fold[:, 0] = self.page_border_color

            # apply curve effect to borders
            curve_frequency = random.randint(self.curve_frequency[0], self.curve_frequency[1])
            for _ in range(curve_frequency):
                border_image_fold = self.random_folding(border_image_fold)
                border_image_fold = np.rot90(self.random_folding(np.rot90(border_image_fold, 3)), 1)

            # draw the rest of borders line
            border_image_fold[-1, :] = self.page_border_color
            border_image_fold[:, -1] = self.page_border_color

            border_image_single = border_image_fold.copy()

            # rotation
            rotated_angle -= rotated_angle_step
            if rotated_angle != 0:
                ysize, xsize = border_image_single.shape[:2]

                # rotate borders
                border_image_single_rotated = rotate_image(border_image_single, rotated_angle, 0)
                nysize, nxsize = border_image_single_rotated.shape[:2]
                border_image_single = border_image_single_rotated[:ysize, nxsize - xsize :]

                # set image as int to assign negative value, so that we can identify the filled area during rotation
                border_image_single = border_image_single.astype("int")

                # check from topleft
                border_image_single = self.set_invalid_value(border_image_single)
                # check from bottomleft
                border_image_single = self.set_invalid_value(np.flipud(border_image_single))
                border_image_single = np.flipud(border_image_single)

                # check from topright
                border_image_single = self.set_invalid_value(np.fliplr(border_image_single))
                # check from bottomrightt
                border_image_single = self.set_invalid_value(np.flipud(border_image_single))
                border_image_single = np.flipud(border_image_single)
                border_image_single = np.fliplr(border_image_single)

            # fill only non negative area
            indices = border_image_single != -1
            # for bottom page, where it didn't shift any value
            if shifted_value_y - border_height == 0 and shifted_value_x - border_width == 0:
                border_image_merged[border_height:, border_width:][indices] = border_image_single[indices]
            # for condition where border width is 0
            elif shifted_value_x - border_width == 0:
                border_image_merged[shifted_value_y : -border_height + shifted_value_y, :][
                    indices
                ] = border_image_single[indices]
            # for condition where border height is 0
            elif shifted_value_y - border_height == 0:
                border_image_merged[:, shifted_value_x : -border_width + shifted_value_x][
                    indices
                ] = border_image_single[indices]
            # for condition where border width and height are not 0
            else:
                border_image_merged[
                    shifted_value_y : -border_height + shifted_value_y,
                    shifted_value_x : -border_width + shifted_value_x,
                ][indices] = border_image_single[indices]

        border_image_merged = cv2.GaussianBlur(border_image_merged, (3, 3), 0)

        # merge each pages into a main page
        if self.same_page_border:
            if border_width == 0:
                border_image_merged[border_height:, :] = image_rotated[border_height:, :]
            elif border_height == 0:
                border_image_merged[:, border_width:] = image_rotated[:, border_width:]
            else:
                border_image_merged[border_height:, border_width:] = image_rotated[border_height:, border_width:]
                border_image_merged[border_height, border_width:] = self.page_border_color
                border_image_merged[border_height:, border_width] = self.page_border_color
        else:
            if border_width == 0:
                border_image_merged[border_height:, :] = image_rotated
            elif border_height == 0:
                border_image_merged[:, border_width:] = image_rotated
            else:
                border_image_merged[border_height:, border_width:] = image_rotated
                border_image_merged[border_height, border_width:] = self.page_border_color
                border_image_merged[border_height:, border_width] = self.page_border_color

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

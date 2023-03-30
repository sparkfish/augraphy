import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

from augraphy.augmentations.brightness import Brightness
from augraphy.augmentations.lib import add_noise as lib_add_noise
from augraphy.augmentations.lib import generate_average_intensity
from augraphy.augmentations.lib import smooth
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class Markup(Augmentation):
    """Uses contours detection to detect text lines and add a smooth text strikethrough, highlight or underline effect.

    :param num_lines_range: Pair of ints determining the number of added markup effect.
    :type num_lines_range: int tuple, optional
    :param markup_length_range: Pair of floats between 0 to 1 , to determine the length of added markup effect.
    :type markup_length_range: float tuple, optional
    :param markup_thickness_range: Pair of ints, to determine the thickness of added markup effect.
    :type markup_thickness_range: int tuple, optional
    :param markup_type: Choice of markup "strikethrough", "highlight", "underline" or "crossed".
    :type markup_type: string
    :param markup_color: BGR color tuple.
    :type markup_color: tuple or string
    :param repetitions: Determine how many time a single markup effect should be drawn.
    :type repetitions: int
    :param large_word_mode: Set true to draw markup on large words, else large word will be ignored.
    :type large_word_mode: boolean
    :param single_word_mode: Set true to draw markup on a single word only.
    :type single_word_mode: boolean
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        num_lines_range=(2, 7),
        markup_length_range=(0.5, 1),
        markup_thickness_range=(1, 3),
        markup_type="random",
        markup_color="random",
        large_word_mode="random",
        single_word_mode=False,
        repetitions=1,
        p=1,
    ):

        super().__init__(p=p)
        self.num_lines_range = num_lines_range
        self.markup_length_range = markup_length_range
        self.markup_thickness_range = markup_thickness_range
        self.markup_type = markup_type
        self.markup_color = markup_color
        self.repetitions = repetitions
        self.large_word_mode = large_word_mode
        self.single_word_mode = single_word_mode

    def __repr__(self):
        return (
            f"Markup(num_lines_range={self.num_lines_range}, markup_length_range={self.markup_length_range}, "
            f"markup_thickness_range={self.markup_thickness_range},  markup_type{self. markup_type}, "
            f"markup_color={self.markup_color}, repetitions={self.repetitions}, "
            f"large_word_mode={self.large_word_mode}, single_word_mode={self.single_word_mode}, p={self.p})"
        )

    def distribute_line(self, starting_point, ending_point, offset):
        """Create smoothed line from the provided starting and ending point.

        :param starting_point: Starting point (x, y) of the line.
        :type starting_point: tuple
        :param ending_point: Ending point (x, y) of the line.
        :type ending_point: tuple
        :param offset: Offset value to randomize point position.
        :type offset: int
        """

        points_count = random.randint(3, 6)  # dividing the line into points
        points_x = np.linspace(starting_point[0], ending_point[0], points_count)
        points_y = [starting_point[1] + random.uniform(-offset, offset) for _ in points_x]
        points = smooth(
            np.column_stack((points_x, points_y)).astype("float"),
            6,
        )  # adding a smoothing effect in points using chaikin's algorithm
        return points

    def _preprocess(self, image):
        """Preprocess image with binarization, dilation and erosion."""
        blurred = cv2.blur(image, (5, 5))
        blurred = blurred.astype("uint8")
        if len(blurred.shape) > 2 and blurred.shape[2] == 3:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        elif len(blurred.shape) > 2 and blurred.shape[2] == 4:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGRA2GRAY)

        _, binarized = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
        )

        # get kernel for dilation
        if self.single_word_mode is False:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            self.markup_length_range = (1, 1)

        # dilating the threshold image to combine horizontal lines
        dilation = cv2.dilate(
            binarized,
            kernel,
            iterations=2,
        )
        dilation = cv2.erode(
            dilation,
            None,
            iterations=1,
        )

        return dilation

    def draw_line(self, p1, p2, markup_mask, markup_thickness, reverse):
        """Draw line across two provided points.

        :param p1: Starting point (x, y) of the line.
        :type p1: tuple
        :param p2: Ending point (x, y) of the line.
        :type p2: tuple
        :param markup_mask: Mask of markup effect.
        :type markup_mask: numpy.array (numpy.uint8)
        :param markup_thickness: Thickness of the line.
        :type markup_thickness: int
        :param reverse: Reverse the order of line points distribution.
        :type reverse: int
        """

        # get min and max of points
        min_x = min(p2[0], p1[0])
        max_x = max(p2[0], p1[0])
        min_y = min(p2[1], p1[1])
        max_y = max(p2[1], p1[1])

        # set point x in ascending or descending order based on direction
        if reverse:
            points_x = [min_x, random.randint(min_x, max_x), max_x]
        else:
            points_x = [max_x, random.randint(min_x, max_x), min_x]
        points_y = [min_y, random.randint(min_y, max_y), max_y]

        # smooth points
        points = smooth(np.column_stack((points_x, points_y)).astype("float"), 6)

        # draw curvy lines
        for (point1_x, point1_y), (point2_x, point2_y) in zip(points[:-1], points[1:]):

            if reverse:
                point1 = (int(point1_x), int(point1_y))
                point2 = (int(point2_x), int(point2_y))
            else:
                point1 = (int(point1_x), int(point1_y))
                point2 = (int(point2_x), int(point2_y))

            markup_mask = cv2.line(
                markup_mask,
                point1,
                point2,
                self.markup_color,
                markup_thickness,
                lineType=cv2.LINE_AA,
            )

    def __call__(self, image, layer=None, force=False):

        # change to 3 channels BGR format
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markup_img = image.copy()
        overlay = markup_img.copy()

        if self.markup_color == "random":
            self.markup_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif self.markup_color == "contrast":
            single_color = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
            self.markup_color = 255 - single_color[0][0]
            self.markup_color = self.markup_color.tolist()

        if self.large_word_mode == "random":
            self.large_word_mode = random.choice([True, False])

        if self.markup_type == "random":
            self.markup_type = random.choice(["strikethrough", "crossed", "underline", "highlight"])

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])
        markup_thickness = random.randint(
            self.markup_thickness_range[0],
            self.markup_thickness_range[1],
        )

        binary_image = self._preprocess(image)

        # Applying dilate operation to connect text lines horizontaly.
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )  # Each line is detected as a contour.

        heights = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            heights.append(h)

        # get average of character height
        bins = np.unique(heights)
        hist, bin_edges = np.histogram(heights, bins=bins, density=False)
        if len(bin_edges) > 1 and np.max(hist) > 20:
            character_height_min = bin_edges[np.argmax(hist)]
            character_height_max = bin_edges[np.argmax(hist) + 1]
            character_height_average = int((character_height_max + character_height_min) / 2)
            height_range = ((character_height_max - character_height_min) / 2) + 1
        else:
            character_height_average = -1
            height_range = -1

        # initialize mask for markup
        markup_mask = np.full_like(overlay, fill_value=255).astype("uint8")

        # shuffle contours to get randomize location to apply augmentation
        if len(contours) > 0:
            contours = list(contours)
            random.shuffle(contours)
        for cnt in contours:
            # adding randomization.
            choice = random.choice([False, True])
            x, y, w, h = cv2.boundingRect(cnt)

            if character_height_average == -1:
                check_height = h > 10
            else:
                check_height = (h > character_height_average - height_range) and (
                    h < character_height_average + height_range
                )

            if not self.large_word_mode:
                conditions = (
                    choice
                    and (w > h * 2)
                    and (w * h < (markup_mask.shape[0] * markup_mask.shape[1]) / 10)
                    and w < int(markup_img.shape[1] / 5)
                    and check_height
                )
            else:
                conditions = check_height

            if conditions:
                if num_lines == 0:
                    break
                num_lines = num_lines - 1
                markup_length = random.uniform(
                    self.markup_length_range[0],
                    self.markup_length_range[1],
                )
                # adjusting width according to markup length
                w = int(w * markup_length)
                # adjusting starting-point according to markup length
                x = int(x + (1 - markup_length) * w)
                # offset to interpolate markup effect up/down
                offset = 6

                # for strikethrough we need center points
                if self.markup_type == "strikethrough":
                    starting_point = [x, int(y + (h / 2))]
                    ending_point = [x + w, int(y + (h / 2))]
                # for corossed-off we need points representing primary diagonal
                elif self.markup_type == "crossed":
                    starting_point = [x, y]
                    ending_point = [x + w, y + h]
                else:
                    # for underline and highlight we need points corresponding to bottom part of text
                    starting_point = [x, y + h]
                    ending_point = [x + w, y + h]

                for i in range(self.repetitions):
                    if self.markup_type == "crossed":

                        ysize, xsize = markup_img.shape[:2]

                        # drawing primary diagonal
                        markup_thickness = random.randint(
                            self.markup_thickness_range[0],
                            self.markup_thickness_range[1],
                        )
                        p1_x = np.clip(
                            starting_point[0] + random.randint(-offset * 5, offset * 5),
                            0,
                            xsize,
                        )
                        p1_y = np.clip(
                            starting_point[1] + +random.randint(-offset * 1, offset * 1),
                            0,
                            ysize,
                        )
                        p2_x = np.clip(
                            ending_point[0] + random.randint(-offset * 5, offset * 5),
                            0,
                            xsize,
                        )
                        p2_y = np.clip(
                            ending_point[1] + +random.randint(-offset * 1, offset * 1),
                            0,
                            ysize,
                        )
                        p1 = (p1_x, p1_y)
                        p2 = (p2_x, p2_y)
                        self.draw_line(p1, p2, markup_mask, markup_thickness, 0)

                        # drawing secondary diagonal
                        markup_thickness = random.randint(
                            self.markup_thickness_range[0],
                            self.markup_thickness_range[1],
                        )
                        p1_x = np.clip(
                            ending_point[0] + random.randint(-offset * 5, offset * 5),
                            0,
                            xsize,
                        )
                        p1_y = np.clip(
                            starting_point[1] + +random.randint(-offset * 1, offset * 1),
                            0,
                            ysize,
                        )
                        p2_x = np.clip(
                            starting_point[0] + random.randint(-offset * 5, offset * 5),
                            0,
                            xsize,
                        )
                        p2_y = np.clip(
                            ending_point[1] + +random.randint(-offset * 1, offset * 1),
                            0,
                            ysize,
                        )
                        p1 = (p1_x, p1_y)
                        p2 = (p2_x, p2_y)
                        self.draw_line(p1, p2, markup_mask, markup_thickness, 1)

                    else:
                        # dividing the line into points to mimic a smoothing effect
                        points_list = self.distribute_line(
                            starting_point,
                            ending_point,
                            offset,
                        ).astype("int")

                        for i in range(len(points_list) - 1):
                            p1 = (int(points_list[i][0]), int(points_list[i][1]))
                            if self.markup_type == "highlight":
                                p2 = (
                                    points_list[i + 1][0],
                                    points_list[i + 1][1] - h,
                                )
                                # A filled rectangle
                                markup_mask = cv2.rectangle(
                                    markup_mask,
                                    p1,
                                    p2,
                                    self.markup_color,
                                    -1,
                                )

                            else:
                                p2 = (
                                    points_list[i + 1][0],
                                    points_list[i + 1][1],
                                )
                                markup_mask = cv2.line(
                                    markup_mask,
                                    p1,
                                    p2,
                                    self.markup_color,
                                    markup_thickness,
                                    lineType=cv2.LINE_AA,
                                )

        # smoothen highlight mask to make it more realistic
        if self.markup_type == "highlight":
            # blur markup mask
            markup_mask = cv2.GaussianBlur(markup_mask, (7, 7), cv2.BORDER_DEFAULT)

            # increase brightness of highlight effect if highlight colour is too dark
            # highlight min intensity ( to appear bright )
            if len(markup_mask.shape) > 2:
                markup_mask_gray = cv2.cvtColor(markup_mask, cv2.COLOR_BGR2GRAY)
            else:
                markup_mask_gray = markup_mask

            # get location of intensity < min intensity
            min_intensity = 200
            y_location, x_location = np.where(markup_mask_gray < min_intensity)

            # if there's location where intensity < min intensity, apply brightness
            if len(y_location) > 0:
                markup_min_intensity = min(markup_mask_gray[y_location, x_location])
                brighten_ratio = abs(markup_min_intensity - min_intensity) / markup_min_intensity
                brighten_min = 1 + brighten_ratio
                brighten_max = 1 + brighten_ratio + 0.5
                brightness = Brightness(brightness_range=(brighten_min, brighten_max))
                markup_mask = brightness(markup_mask)

        else:

            # blur markup mask
            markup_mask = cv2.GaussianBlur(markup_mask, (3, 3), cv2.BORDER_DEFAULT)

            # add scribble similar realistic effect
            markup_mask_copy = markup_mask.copy()

            if len(markup_mask_copy.shape) > 2:
                markup_mask_copy = cv2.cvtColor(markup_mask_copy, cv2.COLOR_BGR2GRAY)
            noise_mask = lib_add_noise(markup_mask_copy, (0.3, 0.5), (32, 128))

            # add noise
            indices = markup_mask_copy < 224
            markup_mask_copy[indices] = noise_mask[indices]

            # add noise
            intensity = random.uniform(0.4, 0.7)
            min_value = 32
            max_value = 128
            ysize, xsize = markup_mask_copy.shape[:2]
            # create random value between min and max
            random_value = np.random.uniform(low=min_value, high=max_value, size=(ysize, xsize)).astype("uint8")
            random_intensity_map = np.random.random((ysize, xsize))
            indices = sobel == 255 and random_intensity_map < intensity
            markup_mask_copy[indices] = random_value[indices]

            markup_mask_copy = cv2.cvtColor(markup_mask_copy, cv2.COLOR_GRAY2BGR)
            markup_mask_copy = cv2.GaussianBlur(markup_mask_copy, (3, 3), 0)

            hsv = cv2.cvtColor(markup_mask_copy.astype("uint8"), cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] += random.randint(0, 128)
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            markup_mask_copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            markup_mask = cv2.multiply(markup_mask_copy, markup_mask, scale=1 / 255)

        # create overlay builder
        overlay_builder = OverlayBuilder(
            "darken",
            markup_mask,
            markup_img,
            1,
            (1, 1),
            "center",
            0,
            alpha=1,
        )
        # overlay image
        markup_img = overlay_builder.build_overlay()

        return markup_img

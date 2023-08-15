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
from augraphy.utilities.inkgenerator import InkGenerator


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
    :param markup_ink: Types of markup ink, choose from "random", "pencil", "pen", "marker" or "highlighter".
    :type markup_ink: string, optional
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
        markup_ink="random",
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
        self.markup_ink = markup_ink
        self.markup_color = markup_color
        self.repetitions = repetitions
        self.large_word_mode = large_word_mode
        self.single_word_mode = single_word_mode

    def __repr__(self):
        return (
            f"Markup(num_lines_range={self.num_lines_range}, markup_length_range={self.markup_length_range}, "
            f"markup_thickness_range={self.markup_thickness_range},  markup_type{self. markup_type}, "
            f"markup_ink={self.markup_ink}, markup_color={self.markup_color}, repetitions={self.repetitions}, "
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

    def draw_line(self, p1, p2, markup_mask, markup_thickness, markup_color, reverse):
        """Draw line across two provided points.

        :param p1: Starting point (x, y) of the line.
        :type p1: tuple
        :param p2: Ending point (x, y) of the line.
        :type p2: tuple
        :param markup_mask: Mask of markup effect.
        :type markup_mask: numpy.array (numpy.uint8)
        :param markup_thickness: Thickness of the line.
        :type markup_thickness: int
        :param markup_color: Color of the line in BGR format.
        :type markup_color: tuple
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
                markup_color,
                markup_thickness,
                lineType=cv2.LINE_AA,
            )

    def __call__(self, image, layer=None, force=False):

        # change to 3 channels BGR format
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markup_image = image.copy()

        if self.markup_color == "random":
            markup_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif self.markup_color == "contrast":
            single_color = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
            markup_color = 255 - single_color[0][0]
            markup_color = markup_color.tolist()
        else:
            markup_color = self.markup_color

        if self.large_word_mode == "random":
            large_word_mode = random.choice([True, False])
        else:
            large_word_mode = self.large_word_mode

        if self.markup_type == "random":
            markup_type = random.choice(["strikethrough", "crossed", "underline", "highlight"])
        else:
            markup_type = self.markup_type

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])

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

        # initialize coordinates of lines
        lines_coordinates = []

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

            if large_word_mode:
                conditions = check_height
            else:
                conditions = (
                    choice
                    and (w > h * 2)
                    and (w * h < (markup_image.shape[0] * markup_image.shape[1]) / 10)
                    and w < int(markup_image.shape[1] / 5)
                    and check_height
                )

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

                # for strikethrough and highlight, we need center points
                if markup_type == "strikethrough" or markup_type == "highlight":
                    starting_point = [x, int(y + (h / 2))]
                    ending_point = [x + w, int(y + (h / 2))]
                # for corossed-off we need points representing primary diagonal
                elif markup_type == "crossed":
                    starting_point = [x, y]
                    ending_point = [x + w, y + h]
                else:
                    # for underline, we need points corresponding to bottom part of text
                    starting_point = [x, y + h]
                    ending_point = [x + w, y + h]

                for i in range(self.repetitions):
                    if markup_type == "crossed":

                        ysize, xsize = markup_image.shape[:2]

                        # primary diagonal
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
                        lines_coordinates.append(np.array([p1, p2]))

                        # secondary diagonal
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
                        lines_coordinates.append(np.array([p1, p2]))

                    else:
                        # dividing the line into points to mimic a smoothing effect
                        points_list = self.distribute_line(
                            starting_point,
                            ending_point,
                            offset,
                        ).astype("int")
                        lines_coordinates.append(points_list)

        # prevent empty lines
        if lines_coordinates:
            # random select ink
            if self.markup_ink == "random":
                markup_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
            else:
                markup_ink = self.markup_ink

            # for highlight, the ink should be thicker
            if self.markup_type == "highlight":
                markup_thickness_range = (self.markup_thickness_range[0] + 5, self.markup_thickness_range[1] + 5)
            else:
                markup_thickness_range = self.markup_thickness_range

            # create ink generator
            ink_generator = InkGenerator(
                ink_type=markup_ink,
                ink_draw_method="lines",
                ink_draw_iterations=(1, 1),
                ink_location="random",
                ink_background=markup_image,
                ink_background_size=None,
                ink_background_color=None,
                ink_color=markup_color,
                ink_min_brightness=1,
                ink_min_brightness_value_range=(150, 200),
                ink_draw_size_range=None,
                ink_thickness_range=markup_thickness_range,
                ink_brightness_change=[0],
                ink_skeletonize=0,
                ink_skeletonize_iterations_range=(1, 1),
                ink_text=None,
                ink_text_font=None,
                ink_text_rotate_range=None,
                ink_lines_coordinates=lines_coordinates,
                ink_lines_stroke_count_range=(1, 1),
            )

            markup_image = ink_generator.generate_ink()

        return markup_image

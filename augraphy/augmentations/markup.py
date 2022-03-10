import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
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
    :param markup_color: bgr color tuple.
    :type markup_color: tuple of ints
    :param repetitions: determines how many time a markup should be drawn
    :type repetitions: int
    :param single_word_mode: set true to draw markup on a single word only
    :type single_word_mode: boolean


    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        num_lines_range=(2, 7),
        markup_length_range=(0.5, 1),
        markup_thickness_range=(1, 3),
        markup_type="strikethrough",
        markup_color=(0, 255, 0),
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
        self.single_word_mode = single_word_mode

    def __repr__(self):
        return (
            f"Markup(num_lines_range={self.num_lines_range}, markup_length_range={self.markup_length_range}, "
            f"markup_thickness_range={self.markup_thickness_range},  markup_type{self. markup_type} p={self.p})"
        )

    def distribute_line(self, starting_point, ending_point, offset):
        points_count = random.randint(3, 10)  # dividing the line into points
        points = np.linspace(starting_point[0], ending_point[0], points_count)
        points = [[int(x), (starting_point[1] + random.randint(-offset, offset))] for x in points]
        points = smooth(points, 6)  # adding a smoothing effect in points using chaikin's algorithm
        return points

    def apply_crossed_off(self, image, p1, p2, offset):
        apply_random_offset_fn = lambda p, offset: p + random.randint(-offset, offset)
        apply_random_offset = np.vectorize(apply_random_offset_fn)
        p1 = tuple(apply_random_offset(p1, offset))
        p2 = tuple(apply_random_offset(p2, offset))
        drawn = cv2.line(image, p1, p2, self.markup_color, self.mar, lineType=cv2.LINE_AA)

    def _preprocess(self, image):
        blurred = cv2.blur(image, (5, 5))
        blurred = blurred.astype("uint8")
        if len(blurred.shape) > 2 and blurred.shape[2] == 3:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        elif len(blurred.shape) > 2 and blurred.shape[2] == 4:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGRA2GRAY)

        ret, thresh1 = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
        )

        if self.single_word_mode is False:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            self.markup_length_range = (1, 1)

        # dilating the threshold image to combine horizontal lines
        dilation = cv2.dilate(
            thresh1,
            kernel,
            iterations=2,
        )
        dilation = cv2.erode(
            dilation,
            None,
            iterations=1,
        )

        return dilation

    def __call__(self, image, layer=None, force=False):
        markup_img = image.copy()
        overlay = markup_img.copy()
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

        # initialize mask for markup
        markup_mask = np.full_like(overlay, fill_value=255).astype("uint8")

        # shuffle contours to get randomize location to apply augmentation
        if len(contours) > 0:
            contours = list(contours)
            random.shuffle(contours)
        for cnt in contours:
            # adding randomization.
            choice = random.choice([False, True])
            if choice:
                x, y, w, h = cv2.boundingRect(cnt)
                # avoiding too small contours (width less  5% of the image width)
                if w < int(markup_img.shape[1] / 5):
                    continue
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
                        # Drawing primary diagonal
                        p1 = (
                            starting_point[0] + random.randint(-offset, offset),
                            starting_point[1] + +random.randint(-offset, offset),
                        )
                        p2 = (
                            ending_point[0] + random.randint(-offset, offset),
                            ending_point[1] + +random.randint(-offset, offset),
                        )
                        markup_mask = cv2.line(
                            markup_mask,
                            p1,
                            p2,
                            self.markup_color,
                            markup_thickness,
                            lineType=cv2.LINE_AA,
                        )
                        # drawing secondary diagonal
                        p1 = (
                            ending_point[0] + random.randint(-offset, offset),
                            starting_point[1] + +random.randint(-offset, offset),
                        )
                        p2 = (
                            starting_point[0] + random.randint(-offset, offset),
                            ending_point[1] + +random.randint(-offset, offset),
                        )
                        markup_mask = cv2.line(
                            markup_mask,
                            p1,
                            p2,
                            self.markup_color,
                            markup_thickness,
                            lineType=cv2.LINE_AA,
                        )

                    else:
                        # dividing the line into points to mimic a smoothing effect
                        points_list = self.distribute_line(starting_point, ending_point, offset)
                        for i in range(len(points_list) - 1):
                            p1 = (int(points_list[i][0]), int(points_list[i][1]))
                            if self.markup_type == "highlight":
                                p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1] - h))
                                # A filled rectangle
                                markup_mask = cv2.rectangle(markup_mask, p1, p2, self.markup_color, -1)

                            else:
                                p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1]))
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
            alpha = 0.5
            markup_mask = cv2.GaussianBlur(markup_mask, (7, 7), cv2.BORDER_DEFAULT)
        else:
            alpha = 1

        # create overlay builder
        overlay_builder = OverlayBuilder(
            "darken",
            markup_mask,
            markup_img,
            1,
            (1, 1),
            "center",
            0,
            alpha=alpha,
        )
        # overlay image
        markup_img = overlay_builder.build_overlay()

        return markup_img

import math
import random
from pathlib import Path

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Markup(Augmentation):
    """
    Uses contours detection to detect text lines and add a smooth text strikethrough effect
    :param num_lines_range: Pair of ints determining the number of lines to add strikethrough
    :type range: int tuple, optional
    :param strikethrough_length_range: Pair of floats between 0 to 1 , to determine the length of strikethrough effect
    :type range: float tuple, optional
    :param strikethrough_thickness_range: Pair of ints, to determine the thickness of strikethrough line
    :type range: int tuple, optional
    :param marktup_type: choice of markup "strikethrough" or "underline"
    :type choice: string
    :param markup_color: rgb color tuple
    :type tuple: int
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        num_lines_range=(2, 7),
        markup_length_range=(0.5, 1),
        markup_thickness_range=(1, 3),
        markup_type="strikethrough",
        markup_color=(0, 0, 0),
        p=0.5,
    ):

        super().__init__(p=p)
        self.layer = layer
        self.num_lines_range = num_lines_range
        self.markup_length_range = markup_length_range
        self.markup_thickness_range = markup_thickness_range
        self.markup_type = markup_type
        self.markup_color = markup_color

    def __repr__(self):
        return (
            f"Markup(layer={self.layer}, num_lines_range={self.num_lines_range}, markup_length_range={self.markup_length_range}, "
            f"markup_thickness_range={self.markup_thickness_range},  markup_type{self. markup_type} p={self.p})"
        )

    def __call__(self, data, force=False):
        image = data[self.layer][-1].result.copy()

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])
        markup_thickness = random.randint(
            self.markup_thickness_range[0],
            self.markup_thickness_range[1],
        )
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
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilation = cv2.dilate(
            thresh1,
            rect_kernel,
            iterations=1,
        )  # Applying dilate operation to connect text lines horizontaly.
        contours, hierarchy = cv2.findContours(
            dilation,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )  # Each line is detected as a contour.
        markup_img = image.copy()
        for cnt in contours:
            choice = random.choice([False, True])  # adding randomization.
            if choice:
                x, y, w, h = cv2.boundingRect(cnt)

                if w < int(markup_img.shape[1] / 5):  # avoiding too small contours (width less  20% of the image width)
                    continue
                if num_lines == 0:
                    break
                num_lines = num_lines - 1
                markup_length = random.uniform(
                    self.markup_length_range[0],
                    self.markup_length_range[1],
                )

                w = int(w * markup_length)  # adjusting width according to markup length
                x = int(x + (1 - markup_length) * w)  # adjusting starting-point according to markup length
                offset = 6
                if self.markup_type == "strikethrough":  # for strikethrough we need center points
                    mid_start = [x, int(y + (h / 2))]
                    mid_end = [x + w, int(y + (h / 2))]

                else:  # for underline we need points corresponding to bottom part of text
                    mid_start = [x, y + h]
                    mid_end = [x + w, y + h]

                points_count = random.randint(3, 10)  # dividing the line into points
                points = np.linspace(mid_start[0], mid_end[0], points_count)
                points_list = [[int(x), (mid_start[1] + random.randint(-offset, offset))] for x in points]
                points_list = smooth(points_list, 8)  # adding a smoothing effect in points using chaikin's algorithm
                for i in range(len(points_list) - 1):
                    p1 = (int(points_list[i][0]), int(points_list[i][1]))
                    p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1]))
                    markup_img = cv2.line(
                        markup_img,
                        p1,
                        p2,
                        self.markup_color,
                        markup_thickness,
                        lineType=cv2.LINE_AA,
                    )

        data[self.layer].append(AugmentationResult(self, markup_img))

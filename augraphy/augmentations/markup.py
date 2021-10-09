import math
import random
from pathlib import Path

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
from augraphy.base.augmentation import Augmentation


class Markup(Augmentation):
    """
    Uses contours detection to detect text lines and add a smooth text strikethrough effect
    :param num_lines_range: Pair of ints determining the number of lines to add strikethrough
    :type range: int tuple, optional
    :param strikethrough_length_range: Pair of floats between 0 to 1 , to determine the length of strikethrough effect
    :type range: float tuple, optional
    :param strikethrough_thickness_range: Pair of ints, to determine the thickness of strikethrough line
    :type range: int tuple, optional
    :param markup_type: choice of markup "strikethrough", "highlight" or "underline"
    :type choice: string
    :param markup_color: bgr color tuple
    :type tuple: int
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
        p=1,
    ):

        super().__init__(p=p)
        self.num_lines_range = num_lines_range
        self.markup_length_range = markup_length_range
        self.markup_thickness_range = markup_thickness_range
        self.markup_type = markup_type
        self.markup_color = markup_color

    def __repr__(self):
        return (
            f"Markup(num_lines_range={self.num_lines_range}, markup_length_range={self.markup_length_range}, "
            f"markup_thickness_range={self.markup_thickness_range},  markup_type{self. markup_type} p={self.p})"
        )

    def __call__(self, image, layer=None, force=False):
        markup_img = image.copy()
        overlay = markup_img.copy()
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
        )
        # Applying dilate operation to connect text lines horizontaly.
        contours, hierarchy = cv2.findContours(
            dilation,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )  # Each line is detected as a contour.

        for cnt in contours:
            # adding randomization.
            choice = random.choice([False, True])
            if choice:
                x, y, w, h = cv2.boundingRect(cnt)
                # avoiding too small contours (width less  20% of the image width)
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
                # Transparency factor.
                alpha = 0.5

                # for strikethrough we need center points
                if self.markup_type == "strikethrough":
                    starting_point = [x, int(y + (h / 2))]
                    ending_point = [x + w, int(y + (h / 2))]
                else:
                    # for underline and highlight we need points corresponding to bottom part of text
                    starting_point = [x, y]
                    ending_point = [x + w, y]

                # dividing the line into points
                points_count = random.randint(3, 10)
                points = np.linspace(starting_point[0], ending_point[0], points_count)
                points_list = [[int(x), (starting_point[1] + random.randint(-offset, offset))] for x in points]
                # adding a smoothing effect in points using chaikin's algorithm
                points_list = smooth(points_list, 8)

                for i in range(len(points_list) - 1):
                    p1 = (int(points_list[i][0]), int(points_list[i][1]))
                    if self.markup_type == "highlight":
                        p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1] + h))
                        # A filled rectangle
                        overlay = cv2.rectangle(overlay, p1, p2, self.markup_color, -1)
                    else:
                        p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1]))
                        markup_img = cv2.line(
                            markup_img,
                            p1,
                            p2,
                            self.markup_color,
                            markup_thickness,
                            lineType=cv2.LINE_AA,
                        )

        if self.markup_type == "highlight":
            markup_img = cv2.addWeighted(overlay, alpha, markup_img, 1 - alpha, 0)

        return markup_img

import math
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Strikethrough(Augmentation):

    """
    Uses contours detection to detect text lines and add a smooth text strikethrough effect
    :param num_lines_range: Pair of ints determining the number of lines to add strikethrough
    :type range: int tuple, optional
    :param strikethrough_length_range: Pair of floats between 0 to 1 , to determine the length of strikethrough effect
    :type range: float tuple, optional
    :param strikethrough_thickness_range: Pair of ints, to determine the thickness of strikethrough line
    :type range: int tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        num_lines_range=(2, 7),
        strikethrough_length_range=(0.5, 1),
        strikethrough_thickness_range=(1, 3),
        p=0.5,
    ):

        super().__init__(p=p)
        self.num_lines_range = num_lines_range
        self.strikethrough_length_range = strikethrough_length_range
        self.strikethrough_thickness_range = strikethrough_thickness_range

    def __repr__(self):
        return (
            f"Strikethrough(num_lines_range={self.num_lines_range}, strikethrough_length_range={self.strikethrough_length_range}, "
            f"strikethrough_thickness_range={self.strikethrough_thickness_range} p={self.p})"
        )

    def chaikin(self, points):
        path = [points[0]]
        percent = 0.25
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            new_p0 = (p0[0] + dx * percent, p0[1] + dy * percent)
            new_p1 = (p0[0] + dx * (1 - percent), p0[1] + dy * (1 - percent))
            path.append(new_p0)
            path.append(new_p1)
        path.append(points[-1])
        return path

    def smooth(self, points, iter):
        for i in range(iter):
            points = self.chaikin(points)
        return points

    def __call__(self, data, force=False):
        image = data["ink"][-1].result.copy()

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])
        strikethrough_thickness = random.randint(
            self.strikethrough_thickness_range[0],
            self.strikethrough_thickness_range[1],
        )
        image = cv2.blur(image, (5, 5))
        image = image.astype("uint8")
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        ret, thresh1 = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
        )
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(
            dilation,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        strikethrough_img = image.copy()
        for cnt in contours:
            choice = random.choice([False, True])
            if choice:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 100:
                    continue
                if num_lines == 0:
                    break
                num_lines -= 1
                strikethrough_length = random.uniform(
                    self.strikethrough_length_range[0],
                    self.strikethrough_length_range[1],
                )
                w = int(w * strikethrough_length)
                x = int(x + (1 - strikethrough_length) * w)
                mid_start = [x, int(y + (h / 2))]
                mid_end = [x + w, int(y + (h / 2))]
                points_count = random.randint(3, 10)
                points = np.linspace(mid_start[0], mid_end[0], points_count)
                points_list = [[int(x), (mid_start[1] + random.randint(-6, 6))] for x in points]
                points_list = self.smooth(points_list, 8)
                for i in range(len(points_list) - 1):
                    p1 = (int(points_list[i][0]), int(points_list[i][1]))
                    p2 = (int(points_list[i + 1][0]), int(points_list[i + 1][1]))
                    strikethrough_img = cv2.line(
                        strikethrough_img,
                        p1,
                        p2,
                        (0, 0, 0),
                        strikethrough_thickness,
                        lineType=cv2.LINE_AA,
                    )

        data["ink"].append(AugmentationResult(self, strikethrough_img))

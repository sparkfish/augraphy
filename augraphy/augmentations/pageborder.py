import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class PageBorder(Augmentation):
    """Add border effect to sides of input image.

    :param side: One of the four sides of page i:e top,right,left,bottom.
                By default it is "left"
    :type side: string , optional
    :param width_range: Pair of ints determining the range from of the page border
    :type width_range: tuple, optional
    :param pages: An integer determining the number of page shadows in the border
    :type pages: int , optional
    :param noise_intensity_range: A pair of floats determining the intensity of
                                  noise being applied around the borders.
    :type noise_intensity_range: tuple , optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        side="random",
        width_range=(5, 30),
        pages=None,
        noise_intensity_range=(0.2, 0.5),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.side = side
        self.width_range = width_range
        self.pages = pages
        self.noise_intensity_range = noise_intensity_range

    def __repr__(self):
        return f"PageBorder(width_range={self.width_range}, pages={self.pages}, noise_intensity_range={self.noise_intensity_range}, p={self.p})"

    def add_corner_noise(self, border, intensity=0.2):
        ksize = (5, 5)
        blur = cv2.blur(border, ksize)
        edge = cv2.Canny(blur, 100, 200)
        Y, X = edge.shape
        idx_list = np.where(edge == 255)
        for i in range(len(idx_list[0])):
            x = idx_list[0][i]
            y = idx_list[1][i]

            reps = random.randint(1, 3)
            for i in range(reps):
                if random.random() < intensity:
                    d = int(random.uniform(1, 5))
                    border[x, min(X - 1, y + d)] = 0
                    d = int(random.uniform(1, 5))
                    border[x, max(0, y - d)] = 0
        return border

    def create_border(
        self,
        channel,
        border_width,
        border_height,
        num_pages=None,
        noise_intensity=0.2,
    ):

        pad = 0
        if channel > 2:
            border = np.ones((border_height, border_width + pad, channel))
        else:
            border = np.ones((border_height, border_width + pad))
        border = border * 255
        color = (0, 0, 0)
        if num_pages is None:
            num_pages = random.randint(2, 7)

        for x in np.linspace(border_width, 0, num_pages):
            x = int(x)
            e = (
                border_width
                if x == border_width
                else np.random.randint(
                    int(border_width - (border_width / 2)),
                    border_width,
                )
            )
            start_point = (x, 0)
            end_point = (e, border_height)
            thickness = np.random.choice([2, 3, 4])
            border = cv2.line(border, start_point, end_point, color, thickness)
        border = self.add_corner_noise(np.uint8(border), noise_intensity)
        border = cv2.blur(border, (3, 3))
        return border

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            noise_intensity = random.uniform(
                self.noise_intensity_range[0],
                self.noise_intensity_range[1],
            )
            border_width = random.randint(self.width_range[0], self.width_range[1])

            height, width = image.shape[:2]
            if len(image.shape) > 2:
                channel = image.shape[2]
            else:
                channel = 1

            if self.side == "random":
                side = random.choice(["left", "right", "top", "bottom"])
            else:
                side = self.side

            if side == "left":
                border = self.create_border(
                    channel,
                    border_width,
                    height,
                    self.pages,
                    noise_intensity,
                )
                # print(image.shape,border.shape)
                image = np.hstack((border, image))
            elif side == "right":
                border = self.create_border(
                    channel,
                    border_width,
                    height,
                    self.pages,
                    noise_intensity,
                )
                image = np.hstack((image, np.fliplr(border)))
            elif side == "top":
                border = self.create_border(
                    channel,
                    border_width,
                    width,
                    self.pages,
                    noise_intensity,
                )
                image = np.vstack((cv2.rotate(border, cv2.ROTATE_90_CLOCKWISE), image))
            elif side == "bottom":
                border = self.create_border(
                    channel,
                    border_width,
                    width,
                    self.pages,
                    noise_intensity,
                )
                image = np.vstack(
                    (image, (cv2.rotate(border, cv2.ROTATE_90_COUNTERCLOCKWISE))),
                )

            return image

import numpy as np
import random

import cv2
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class PageBorderAugmentation(Augmentation):
    """

            :param side: th
            :param width_range:
            :param pages:
            :param noise_intensity_range:
            :param p:
            """

    def __init__(
        self,
        side="left",
        width_range=(5,30),
        pages=None,
        noise_intensity_range=(0.2 , 0.5),
        p=0.5
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.side=side
        self.width_range=width_range
        self.pages=pages
        self.noise_intensity_range=noise_intensity_range

    def __repr__(self):
        return f"PageBorderAugmentation(width_range={self.width_range}, pages={self.pages}, noise_intensity_range={self.noise_intensity_range}, p={self.p})"

    def add_corner_noise(self,border, intensity=0.2):
        ksize = (5, 5)
        blur = cv2.blur(border, ksize)
        edge = cv2.Canny(blur, 100, 200)
        Y, X = (edge.shape)
        idx_list = np.where(edge == 255)
        for i in range(len(idx_list[0])):
            x = idx_list[0][i]
            y = idx_list[1][i]

            reps = random.randint(1, 3)
            for i in range(reps):
                if random.random() < intensity:
                    d = int(random.uniform(1, 5))
                    border[x, min(X - 1, y + d), :] = (0, 0, 0)
                    d = int(random.uniform(1, 5))
                    border[x, max(0, y - d), :] = (0, 0, 0)
        return border

    def create_border(self,border_width, border_height, num_pages=None, noise_intensity=0.2):

        pad = 0
        border = np.ones((border_height, border_width + pad, 3))
        border = border * 255
        color = (0, 0, 0)
        if num_pages is None:
            num_pages = random.randint(2, 7)

        for x in np.linspace(border_width, 0, num_pages):
            x = int(x)
            e = border_width if x == border_width else np.random.randint(int(border_width - (border_width / 2)),
                                                                         border_width)
            start_point = (x, 0)
            end_point = (e, border_height)
            thickness = np.random.choice([2, 3, 4])
            border = cv2.line(border, start_point, end_point, color, thickness)
        border = self.add_corner_noise(np.uint8(border), noise_intensity)
        border = cv2.blur(border, (3, 3))
        return border
import random

import numpy as np

from augraphy.augmentations.lib import warp_fold_left_side
from augraphy.augmentations.lib import warp_fold_right_side
from augraphy.base.augmentation import Augmentation


class Folding(Augmentation):
    """Emulates folding effect from perspective transformation

    :param fold_x: X coordinate of the folding effect.
    :type fold_x: int, optional
    :param fold_deviation: Deviation (in pixels) of provided X coordinate location.
    :type fold_deviation: tuple, optional
    :param fold count: Number of applied foldings
    :type fold_count: int, optional
    :param fold_noise: Level of noise added to folding area. Range from
                        value of 0 to 1.
    :type fold_noise: float, optional
    :param gradient_width: Tuple (min, max) Measure of the space affected
                            by fold prior to being warped (in units of
                            percentage of width of page)
    :type gradient_width: tuple, optional
    :param gradient_height: Tuple (min, max) Measure of depth of fold (unit
                            measured as percentage page height)
    :type gradient_height: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        fold_x=None,
        fold_deviation=(0, 0),
        fold_count=2,
        fold_noise=0.1,
        gradient_width=(0.1, 0.2),
        gradient_height=(0.01, 0.02),
        p=1,
    ):
        super().__init__(p=p)
        self.fold_x = fold_x
        self.fold_deviation = fold_deviation
        self.fold_count = fold_count
        self.fold_noise = fold_noise
        self.gradient_width = gradient_width
        self.gradient_height = gradient_height

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Folding(fold_x={self.fold_x}, fold_deviation={self.fold_deviation}, fold_count={self.fold_count}, fold_noise={self.fold_noise}, gradient_width={self.gradient_width}, gradient_height={self.gradient_height},p={self.p})"

    # Apply perspective transform 2 times and get single folding effect
    def apply_folding(
        self,
        img,
        ysize,
        xsize,
        gradient_width,
        gradient_height,
        fold_noise,
    ):

        min_fold_x = min(np.ceil(gradient_width[0] * xsize), xsize).astype("int")
        max_fold_x = min(np.ceil(gradient_width[1] * xsize), xsize).astype("int")
        fold_width_one_side = int(
            random.randint(min_fold_x, max_fold_x) / 2,
        )  # folding width from left to center of folding, or from right to center of folding

        # test for valid folding center line
        if (xsize - fold_width_one_side - 1) < (fold_width_one_side + 1):
            print("Folding augmentation is not applied, please increase image size")
            return img

        # center of folding
        if self.fold_x is None:

            fold_x = random.randint(
                fold_width_one_side + 1,
                xsize - fold_width_one_side - 1,
            )
        else:
            deviation = random.randint(
                self.fold_deviation[0],
                self.fold_deviation[1],
            ) * random.choice([-1, 1])
            fold_x = min(
                max(self.fold_x + deviation, fold_width_one_side + 1),
                xsize - fold_width_one_side - 1,
            )

        fold_y_shift_min = min(np.ceil(gradient_height[0] * ysize), ysize).astype("int")
        fold_y_shift_max = min(np.ceil(gradient_height[1] * ysize), ysize).astype("int")
        fold_y_shift = random.randint(
            fold_y_shift_min,
            fold_y_shift_max,
        )  # y distortion in folding (support positive y value for now)

        if (fold_width_one_side != 0) and (fold_y_shift != 0):
            img_fold_l = warp_fold_left_side(
                img,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
            )
            img_fold_r = warp_fold_right_side(
                img_fold_l,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
            )
            return img_fold_r
        else:
            if fold_width_one_side == 0:
                print(
                    "Folding augmentation is not applied, please increase gradient width or image size",
                )
            else:
                print(
                    "Folding augmentation is not applied, please increase gradient height or image size",
                )
            return img

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # get image dimension
            if len(image.shape) > 2:
                ysize, xsize, _ = image.shape
            else:
                ysize, xsize = image.shape

            # apply folding multiple times
            image_fold = image.copy()
            for _ in range(self.fold_count):
                image_fold = self.apply_folding(
                    image_fold,
                    ysize,
                    xsize,
                    self.gradient_width,
                    self.gradient_height,
                    self.fold_noise,
                )

            return image_fold

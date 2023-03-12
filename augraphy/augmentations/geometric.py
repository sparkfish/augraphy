import random

import cv2
import numpy as np

from augraphy.augmentations.lib import rotate_image
from augraphy.base.augmentation import Augmentation


class Geometric(Augmentation):
    """Applies basic geometric transformations such as resizing, flips and rotation.

    :param scale: Pair of floats determining new scale of image.
    :type scale: tuple, optional
    :param translation: Pair of ints determining x and y translation value.
    :type translation: tuple, optional
    :param fliplr: Flag to flip image in left right direction.
    :type fliplr: int, optional
    :param flipud: Flag to flip image in up down direction.
    :type flipud: int, optional
    :param crop: Tuple of 4 (x0, y0, xn, yn) to crop section of image.
    :type crop: tuple, optional
    :param rotate_range: Pair of ints determining the range from which to sample
           the image rotation.
    :type rotate_range: tuple, optional
    :param randomize: Flag to apply random geometric transformations.
    :param randomize: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scale=(1, 1),
        translation=(0, 0),
        fliplr=0,
        flipud=0,
        crop=(),
        rotate_range=(0, 0),
        randomize=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.scale = scale
        self.translation = translation
        self.fliplr = fliplr
        self.flipud = flipud
        self.crop = crop
        self.rotate_range = rotate_range
        self.randomize = randomize

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Geometry(scale={self.scale}, translation={self.translation}, fliplr={self.fliplr}, flipud={self.flipud}, crop={self.crop}, rotate_range={self.rotate_range}, randomize={self.randomize}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            if self.randomize:
                # randomize scale
                scale = (random.uniform(0.5, 1), random.uniform(1, 1.5))

                # randomize translation value
                ysize, xsize = image.shape[:2]
                self.translation = (random.randint(0, int(xsize * 0.1)), random.randint(0, int(ysize * 0.1)))

                # randomize flip
                self.fliplr = random.choice([0, 1])
                self.flipud = random.choice([0, 1])

                # randomize crop
                cx1 = random.randint(0, int(xsize / 5))
                cx2 = random.randint(int(xsize / 2), xsize - 1)
                cy1 = random.randint(0, int(ysize / 5))
                cy2 = random.randint(int(ysize / 2), ysize - 1)
                self.crop = (cx1, cy1, cx2, cy2)

                # randomize rotate
                self.rotate_range = (-10, 10)

            # crop image
            if self.crop:
                # make sure there's only 4 inputs, x0, y0, xn, yn
                if len(self.crop) == 4:
                    ysize, xsize = image.shape[:2]
                    xstart, ystart, xend, yend = self.crop
                    # when value is set to -1, it takes image size
                    if yend == -1:
                        yend = ysize
                    if xend == -1:
                        xend = xsize
                    # condition to make sure cropping range is valid
                    check_y = yend > ystart and ystart >= 0
                    check_x = xend > xstart and xstart >= 0
                    # crop image
                    if check_y and check_x:
                        image = image[ystart:yend, xstart:xend]

            # resize based on scale
            if self.scale[1] > 0 and self.scale[0] > 0:
                scale = random.uniform(self.scale[0], self.scale[1])
                if scale > 0:
                    new_width = int(image.shape[1] * scale)
                    new_height = int(image.shape[0] * scale)
                    new_size = (new_width, new_height)
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            # translate image based on translation value
            if self.translation[0] != 0 or self.translation[1] != 0:

                image_new = np.full_like(image, fill_value=255).astype("uint8")
                offset_x = self.translation[0]
                offset_y = self.translation[1]

                # x translation
                if offset_x > 0:
                    image_new[:, offset_x:] = image[:, :-offset_x]
                    image = image_new
                elif offset_x < 0:
                    image_new[:, :offset_x] = image[:, abs(offset_x) :]
                    image = image_new

                image_new = np.full_like(image, fill_value=255).astype("uint8")

                # y translation
                if offset_y > 0:
                    image_new[offset_y:, :] = image[:-offset_y, :]
                    image = image_new
                elif offset_y < 0:
                    image_new[:offset_y, :] = image[abs(offset_y) :, :]
                    image = image_new

            # flip left right
            if self.fliplr:
                image = np.fliplr(image)

            # flip up down
            if self.flipud:
                image = np.flipud(image)

            # generate random angle
            if (self.rotate_range[0] != 0) | (self.rotate_range[1] != 0):
                angle = random.randint(self.rotate_range[0], self.rotate_range[1])
            else:
                angle = 0
            # rotate image
            if angle != 0:
                image = rotate_image(image, angle)

            return image

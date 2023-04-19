import random

import cv2
import numpy as np

from augraphy.augmentations.lib import rotate_image
from augraphy.base.augmentation import Augmentation


class Geometric(Augmentation):
    """Applies basic geometric transformations such as resizing, flips and rotation.

    :param scale: Pair of floats determining new scale of image.
    :type scale: tuple, optional
    :param translation: Pair of values determining x and y translation value.
            The translation value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
            x (int) = image width  * x (float and 0.0 - 1.0);
            y (int) = image height * y (float and 0.0 - 1.0)
    :type translation: tuple, optional
    :param fliplr: Flag to flip image in left right direction.
    :type fliplr: int, optional
    :param flipud: Flag to flip image in up down direction.
    :type flipud: int, optional
    :param crop: Tuple of 4 (x0, y0, xn, yn) to crop section of image.
             The value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
             x0 (int) = image width  * x0 (float and 0.0 - 1.0);
             y0 (int) = image height * y0 (float and 0.0 - 1.0);
             xn (int) = image width  * xn (float and 0.0 - 1.0);
             yn (int) = image height * yn (float and 0.0 - 1.0)
    :type crop: tuple, optional
    :param rotate_range: Pair of ints determining the range from which to sample
           the image rotation.
    :type rotate_range: tuple, optional
    :param randomize: Flag to apply random geometric transformations.
    :type randomize: int, optional
    :param padding: Padding amount on each (left, right, top, bottom) side.
            The padding amount will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
            left   (int) = image width  * left   (float and 0.0 - 1.0);
            right  (int) = image height * right  (float and 0.0 - 1.0);
            top    (int) = image width  * top    (float and 0.0 - 1.0);
            bottom (int) = image height * bottom (float and 0.0 - 1.0)
    :type padding: list, optional
    :param padding_type: Padding methods, select from fill,duplicate and mirror.
    :type paddng_type: string, optional
    :param padding_value: Padding value (in BGR) for fill padding method.
    :type paddng_value: tuple, optional
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
        padding=[0, 0, 0, 0],
        padding_type="fill",
        padding_value=(255, 255, 255),
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
        self.padding = padding
        self.padding_type = padding_type
        self.padding_value = padding_value

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Geometry(scale={self.scale}, translation={self.translation}, fliplr={self.fliplr}, flipud={self.flipud}, crop={self.crop}, rotate_range={self.rotate_range}, padding={self.padding}, padding_type={self.padding_type}, padding_value={self.padding_value}, randomize={self.randomize}, p={self.p})"

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

                # randomize padding
                self.padding = [
                    random.randint(0, int(xsize / 5)),
                    random.randint(0, int(xsize / 5)),
                    random.randint(0, int(ysize / 5)),
                    random.randint(0, int(ysize / 5)),
                ]
                self.padding_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self.padding_typee = random.choice(["fill", "mirror", "duplicate"])

            # crop image
            if self.crop:
                # make sure there's only 4 inputs, x0, y0, xn, yn
                if len(self.crop) == 4:
                    ysize, xsize = image.shape[:2]
                    xstart, ystart, xend, yend = self.crop

                    # when value is float and in between 0-1, scale it with image size
                    if xstart >= 0 and xstart <= 1 and isinstance(xstart, float):
                        xstart = int(xstart * xsize)
                    if ystart >= 0 and ystart <= 1 and isinstance(ystart, float):
                        ystart = int(ystart * ysize)
                    if xend >= 0 and xend <= 1 and isinstance(xend, float):
                        xend = int(xend * xsize)
                    if yend >= 0 and yend <= 1 and isinstance(yend, float):
                        yend = int(yend * ysize)

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

            if any(self.padding):

                # convert from rgb to grayscale using their average
                if len(image.shape) < 3:
                    self.padding_value = np.mean(self.padding_value)

                # padding on left side
                if self.padding[0] > 0:
                    # get image size
                    ysize, xsize = image.shape[:2]
                    # convert percentage into pixel amount
                    if self.padding[0] <= 1 and isinstance(self.padding[0], float):
                        self.padding[0] = int(self.padding[0] * xsize)

                    # different padding shape for grayscale and colored image
                    if len(image.shape) > 2:
                        padding_shape = (ysize, self.padding[0], image.shape[2])
                    else:
                        padding_shape = (ysize, self.padding[0])
                    # create the padding image
                    if self.padding_type == "duplicate":
                        image_padding = image[:, -self.padding[0] :].copy()
                    elif self.padding_type == "mirror":
                        image_padding = np.fliplr(image[:, : self.padding[0]].copy())
                    else:
                        image_padding = np.full(padding_shape, fill_value=self.padding_value, dtype="uint8")
                    # combine padding image and original image
                    image = np.concatenate([image_padding, image], axis=1)

                # padding on right side
                if self.padding[1] > 0:
                    # get image size
                    ysize, xsize = image.shape[:2]
                    # convert percentage into pixel amount
                    if self.padding[1] <= 1 and isinstance(self.padding[1], float):
                        self.padding[1] = int(self.padding[1] * xsize)

                    # different padding shape for grayscale and colored image
                    if len(image.shape) > 2:
                        padding_shape = (ysize, self.padding[1], image.shape[2])
                    else:
                        padding_shape = (ysize, self.padding[1])
                    # create the padding image
                    if self.padding_type == "duplicate":
                        image_padding = image[:, : self.padding[1]].copy()
                    elif self.padding_type == "mirror":
                        image_padding = np.fliplr(image[:, -self.padding[1] :].copy())
                    else:
                        image_padding = np.full(padding_shape, fill_value=self.padding_value, dtype="uint8")
                    # combine padding image and original image
                    image = np.concatenate([image, image_padding], axis=1)

                # padding on top side
                if self.padding[2] > 0:
                    # get image size
                    ysize, xsize = image.shape[:2]
                    # convert percentage into pixel amount
                    if self.padding[2] <= 1 and isinstance(self.padding[2], float):
                        self.padding[2] = int(self.padding[2] * ysize)

                    # different padding shape for grayscale and colored image
                    if len(image.shape) > 2:
                        padding_shape = (self.padding[2], xsize, image.shape[2])
                    else:
                        padding_shape = (self.padding[2], xsize)
                    # create the padding image
                    if self.padding_type == "duplicate":
                        image_padding = image[-self.padding[2] :, :].copy()
                    elif self.padding_type == "mirror":
                        image_padding = np.flipud(image[: self.padding[2], :].copy())
                    else:
                        image_padding = np.full(padding_shape, fill_value=self.padding_value, dtype="uint8")
                    # combine padding image and original image
                    image = np.concatenate([image_padding, image], axis=0)

                # padding on bottom side
                if self.padding[3] > 0:
                    # get image size
                    ysize, xsize = image.shape[:2]
                    # convert percentage into pixel amount
                    if self.padding[3] <= 1 and isinstance(self.padding[3], float):
                        self.padding[3] = int(self.padding[3] * ysize)

                    # different padding shape for grayscale and colored image
                    if len(image.shape) > 2:
                        padding_shape = (self.padding[3], xsize, image.shape[2])
                    else:
                        padding_shape = (self.padding[3], xsize)
                    # create the padding image
                    if self.padding_type == "duplicate":
                        image_padding = image[: self.padding[3], :].copy()
                    elif self.padding_type == "mirror":
                        image_padding = np.flipud(image[-self.padding[3] :, :].copy())
                    else:
                        image_padding = np.full(padding_shape, fill_value=self.padding_value, dtype="uint8")
                    # combine padding image and original image
                    image = np.concatenate([image, image_padding], axis=0)

            # resize based on scale
            # remove negative value (if any)
            self.scale = list(self.scale)
            self.scale[0] = abs(self.scale[0])
            self.scale[1] = abs(self.scale[1])
            if self.scale[1] != 1 and self.scale[0] != 1:
                scale = random.uniform(self.scale[0], self.scale[1])
                if scale > 0:
                    new_width = int(image.shape[1] * scale)
                    new_height = int(image.shape[0] * scale)
                    new_size = (new_width, new_height)
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            # translate image based on translation value
            if self.translation[0] != 0 or self.translation[1] != 0:

                ysize, xsize = image.shape[:2]
                if self.translation[0] <= 1 and self.translation[0] >= -1 and isinstance(self.translation[0], float):
                    self.translation = list(self.translation)
                    self.translation[0] = int(self.translation[0] * xsize)
                if self.translation[1] <= 1 and self.translation[1] >= -1 and isinstance(self.translation[1], float):
                    self.translation = list(self.translation)
                    self.translation[1] = int(self.translation[1] * ysize)

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

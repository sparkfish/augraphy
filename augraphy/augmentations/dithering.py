import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Dithering(Augmentation):
    """
    Applies Ordered or Floyd Steinberg dithering to the input image.

    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param dither: Types of dithering, ordered or Floyd Steinberg dithering.
    :type dither: string, optional
    :param order: Order number for ordered dithering.
    :type order: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        dither="ordered",
        order=5,
        p=0.5,
    ):
        super().__init__(p=p)
        self.layer = layer
        self.dither = dither
        self.order = order

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Dithering({self.layer}, dither={self.dither}, p={self.p})"

    # Apply Floyd Steinberg dithering algorithm
    def apply_Floyd_Steinberg(self, image, ysize, xsize):
        for y in range(1, ysize - 1):
            for x in range(1, xsize - 1):
                old_pixel = image[y, x]
                new_pixel = 255 * np.floor(old_pixel / 128)
                image[y, x] = new_pixel
                quant_error = min(old_pixel - new_pixel, 0)  # remove negative
                image[y, x + 1] += quant_error * (7 / 16)
                image[y + 1, x - 1] += quant_error * (3 / 16)
                image[y + 1, x] += quant_error * (5 / 16)
                image[y + 1, x + 1] += quant_error * (1 / 16)
        return image.astype("uint8")

    # Floyd Steinberg dithering
    def dither_Floyd_Steinberg(self, image):

        if len(image.shape) > 2:  # coloured image
            ysize, xsize, dim = image.shape
            img_dither_fs = image.copy().astype("float")
            for channel_num in range(dim):
                img_dither_fs[:, :, channel_num] = self.apply_Floyd_Steinberg(
                    img_dither_fs[:, :, channel_num],
                    ysize,
                    xsize,
                )
        else:  # grayscale or binary
            ysize, xsize = image.shape
            img_dither_fs = image.copy().astype("float")
            img_dither_fs = self.apply_Floyd_Steinberg(img_dither_fs, ysize, xsize)

        return img_dither_fs.astype("uint8")

    # Apply ordered dithering algorithm
    def apply_Ordered(self, image, ysize, xsize, order, ordered_matrix):
        for y in range(ysize):
            for x in range(xsize):
                oy = y % order
                ox = x % order
                if image[y, x] > ordered_matrix[oy][ox]:
                    image[y, x] = 255
                else:
                    image[y, x] = 0
        return image.astype("uint8")

    # Ordered dithering
    def dither_Ordered(self, image, order=5):

        # create bayer matrix based on the order
        ordered_matrix = self.create_bayer(0, 0, 2 ** (order), 0, 1)
        total_number = len(ordered_matrix) * len(ordered_matrix[0]) - 1
        # quantitize order matrix value
        for y, row in enumerate(ordered_matrix):
            for x, value in enumerate(row):
                ordered_matrix[y][x] = np.floor((value / total_number) * 255)

        if len(image.shape) > 2:  # coloured image
            ysize, xsize, dim = image.shape
            img_dither_ordered = image.copy().astype("float")
            for channel_num in range(dim):
                img_dither_ordered[:, :, channel_num] = self.apply_Ordered(
                    img_dither_ordered[:, :, channel_num],
                    ysize,
                    xsize,
                    order,
                    ordered_matrix,
                )
        else:  # grayscale or binary
            ysize, xsize = image.shape
            img_dither_ordered = image.copy().astype("float")
            img_dither_ordered = self.apply_Ordered(
                img_dither_ordered,
                ysize,
                xsize,
                order,
                ordered_matrix,
            )

        return img_dither_ordered.astype("uint8")

        return img_dither_ordered

    # Adapted from https://github.com/tromero/BayerMatrix
    def create_bayer(self, x, y, size, value, step, matrix=[[]]):
        if matrix == [[]]:
            matrix = [[0 for i in range(size)] for i in range(size)]
        if size == 1:
            matrix[int(y)][int(x)] = value
            return
        half = size / 2
        # subdivide into quad tree and call recursively
        # pattern is TL, BR, TR, BL
        self.create_bayer(x, y, half, value + (step * 0), step * 4, matrix)
        self.create_bayer(
            x + half,
            y + half,
            half,
            value + (step * 1),
            step * 4,
            matrix,
        )
        self.create_bayer(x + half, y, half, value + (step * 2), step * 4, matrix)
        self.create_bayer(x, y + half, half, value + (step * 3), step * 4, matrix)
        return matrix

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result.copy()

            if self.dither == "ordered":
                image_dither = self.dither_Ordered(image, self.order)
            else:
                image_dither = self.dither_Floyd_Steinberg(image)

            data[self.layer].append(AugmentationResult(self, image_dither))

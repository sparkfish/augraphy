import random

import cv2
import numpy as np

from augraphy.augmentations.lib import binary_threshold
from augraphy.base.augmentation import Augmentation


class Faxify(Augmentation):
    """Emulates faxify effect in the image.

    :param scale_range: Pair of ints determining the range from which to
           divide the resolution by.
    :type scale_range: tuple, optional
    :param monochrome: Flag to apply monochrome effect, false to select
           halftone effect.
    :type monochrome: int, optional
    :param monochrome_method: Otsu, Simple or Adaptive method.
    :type monochrome_method: string, optional
    :param monochrome_threshold: Simple binarization threshold value.
    :type monochrome_threshold: int, optional
    :param invert: Invert grayscale value in halftone effect.
    :type invert: int, optional
    :param half_kernel_size: Half size of gaussian kernel for halftone effect.
    :type half_kernel_size: int, optional
    :param angle: Angle of halftone effect.
    :type angle: int, optional
    :param sigma: Sigma value of gaussian kernel in halftone effect.
    :type sigma: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scale_range=(1, 1),
        monochrome=1,
        monochrome_method="Otsu",
        monochrome_threshold=127,
        invert=1,
        half_kernel_size=2,
        angle=45,
        sigma=2,
        p=1,
    ):

        """Constructor method"""
        super().__init__(p=p)
        self.scale_range = scale_range
        self.monochrome = monochrome
        self.monochrome_method = monochrome_method
        self.monochrome_threshold = 127
        self.invert = invert
        self.half_kernel_size = half_kernel_size
        self.angle = angle
        self.sigma = sigma

        self.enable_otsu = 0
        self.enable_simple = 0
        self.enable_adaptive = 0
        if self.monochrome_method == "Otsu":
            self.enable_otsu = 1
        elif self.monochrome_method == "Simple":
            self.enable_simple = 1
        elif self.monochrome_method == "Adaptive":
            self.enable_adaptive = 1

        # at least 1 for half kernel size
        self.half_kernel_size = max(1, self.half_kernel_size)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Faxify(scale_range={self.scale_range}, monochrome={self.monochrome}, monochrome_method={self.monochrome_method}, monochrome_threshold={self.monochrome_threshold}, invert={self.invert}, half_kernel_size={self.half_kernel_size}, angle={self.angle}, sigma={self.sigma}, p={self.p})"

    # rotate image based on the input angle
    def cv_rotate(self, image, angle):

        # image shape
        ysize, xsize = image.shape[:2]
        # center of rotation
        cx, cy = xsize // 2, ysize // 2
        # rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)

        # rotation calculates the cos and sin, taking absolutes of those
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        # find the new x and y bounds
        bound_x = int(ysize * abs_sin + xsize * abs_cos)
        bound_y = int(ysize * abs_cos + xsize * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        M[0, 2] += bound_x / 2 - cx
        M[1, 2] += bound_y / 2 - cy

        # warp and rotate the image
        image_rotated = cv2.warpAffine(image, M, (bound_x, bound_y))

        return image_rotated

    # generate halftone effect
    def generate_halftone(self, image, invert=1, half_kernel_size=2, angle=45, sigma=2):

        # get total width of the kernel
        kernel_size = kernel_size_x = kernel_size_y = 2 * half_kernel_size + 1

        # rotate image based on the angle
        rotated = self.cv_rotate(image, angle)

        # get new rotated image size
        ysize, xsize = rotated.shape[:2]

        # generate gaussian kernel
        image_kernel = np.zeros((kernel_size_x, kernel_size_x))
        image_kernel[half_kernel_size, half_kernel_size] = 1
        gaussian_kernel = cv2.GaussianBlur(
            image_kernel,
            (kernel_size_x, kernel_size_y),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        gaussian_kernel *= 1 / np.max(gaussian_kernel)

        # initialize empty image
        image_halftone = np.zeros((ysize, xsize))

        # generate halftone effect by using (image window average value * gaussian kernel)
        for y in range(0, ysize - kernel_size + 1, kernel_size):
            for x in range(0, xsize - kernel_size + 1, kernel_size):
                image_halftone[y : y + kernel_size, x : x + kernel_size] = (
                    np.mean(rotated[y : y + kernel_size, x : x + kernel_size]) * gaussian_kernel
                )

        # rotate back using negative angle
        image_halftone = self.cv_rotate(image_halftone, -angle)

        # crop the center section of image
        ysize_out, xsize_out = image_halftone.shape[:2]
        ysize_in, xsize_in = image.shape[:2]
        y_start = int((ysize_out - ysize_in) / 2)
        y_end = y_start + ysize_in
        x_start = int((xsize_out - xsize_in) / 2)
        x_end = x_start + xsize_in
        image_halftone = image_halftone[y_start:y_end, x_start:x_end]

        # return image as uint8
        if invert:
            return ((1 - image_halftone) * 255).astype("uint8")
        else:
            return (image_halftone * 255).astype("uint8")

    # convert rgb/bgr to single channel grayscale
    def rgb_to_gray(self, img, invert=1, gray_level=255, max_value=255):

        img_complement = max_value - img

        if len(img.shape) > 2:
            img_gray = np.min(img_complement, axis=2) * (gray_level / max_value)
            img_gray[np.where(np.sum(img, axis=2) == 0)] = max_value  # if there is no color, set it to max value
        else:
            img_gray = img_complement * (gray_level / max_value)
            img_gray[np.where(img == 0)] = max_value  # if there is no color, set it to max value

        if invert:
            return (img_gray / 255).astype("float")
        else:
            return (1 - (img_gray / 255)).astype("float")

    # downscale image based on the provided scale
    def downscale(self, image):

        if len(image.shape) > 2:
            ysize, xsize, dim = image.shape
        else:
            ysize, xsize = image.shape

        scale = random.randint(self.scale_range[0], self.scale_range[1])
        new_size = (xsize // scale, ysize // scale)
        image_downscaled = cv2.resize(image, new_size)

        return image_downscaled

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run() or True:
            image = image.copy()

            # downscale image
            image_downscaled = self.downscale(image)

            # applies monochrome
            if self.monochrome:
                image_out = binary_threshold(
                    image_downscaled,
                    enable_otsu=self.enable_otsu,
                    enable_simple=self.enable_simple,
                    simple_method=cv2.THRESH_BINARY,
                    thres=self.monochrome_threshold,
                    max_value=255,
                    enable_adaptive=self.enable_adaptive,
                    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    block_size=21,
                    C=10,
                )
            # applies halftone
            else:
                image_gray = self.rgb_to_gray(image_downscaled, invert=self.invert)

                image_out = self.generate_halftone(
                    image_gray,
                    self.invert,
                    self.half_kernel_size,
                    self.angle,
                    self.sigma,
                )

            # upscale image
            image_faxify = cv2.resize(image_out, (image.shape[1], image.shape[0]))

            return image_faxify

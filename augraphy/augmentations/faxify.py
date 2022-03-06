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
    :param monochrome: Flag to enable monochrome effect.
    :type monochrome: int, optional
    :param monochrome_method: Monochrome thresholding method.
    :type monochrome_method: string, optional
    :param monochrome_arguments: A dictionary contains argument to monochrome
            thresholding method.
    :type monochrome_arguments: dict, optional
    :param halftone: Flag to enable halftone effect.
    :type halftone: int, optional
    :param invert: Flag to invert grayscale value in halftone effect.
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
        scale_range=None,
        monochrome=None,
        monochrome_method=None,
        monochrome_arguments=None,
        halftone=None,
        invert=None,
        half_kernel_size=None,
        angle=None,
        sigma=None,
        p=1,
    ):

        """Constructor method"""
        super().__init__(p=p)
        self.scale_range = scale_range
        self.monochrome = monochrome
        self.monochrome_method = monochrome_method
        self.monochrome_arguments = monochrome_arguments
        self.halftone = halftone
        self.invert = invert
        self.half_kernel_size = half_kernel_size
        self.angle = angle
        self.sigma = sigma

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Faxify(scale_range={self.scale_range}, monochrome={self.monochrome}, monochrome_method={self.monochrome_method}, monochrome_arguments={self.monochrome_arguments}, halftone={self.halftone}, invert={self.invert}, half_kernel_size={self.half_kernel_size}, angle={self.angle}, sigma={self.sigma}, p={self.p})"

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
    def generate_halftone(self, image, half_kernel_size=2, angle=45, sigma=2):

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

        return image_halftone

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
            return img_gray.astype("float") / 255
        else:
            return 1 - (img_gray.astype("float") / 255)

    # downscale image based on the provided scale
    def downscale(self, image):

        ysize, xsize = image.shape[:2]
        scale = random.randint(self.scale_range[0], self.scale_range[1])
        new_size = (xsize // scale, ysize // scale)
        image_downscaled = cv2.resize(image, new_size)

        return image_downscaled

    # initialize default values based on input image
    def init_default_values(self, image):

        ysize, xsize = image.shape[:2]
        resolution = ysize * xsize
        min_size = min(ysize, xsize)
        internal_resolution = 1500 * 1500

        # scale to internal default testing size
        scale = int(resolution / internal_resolution)

        if self.scale_range is None:
            self.scale_range = (1, 3)

        if self.monochrome is None:
            self.monochrome = 1

        if self.monochrome:
            # randomly select threshold method
            if self.monochrome_method is None:
                all_monochrome_method = [
                    "threshold_li",
                    "threshold_local",
                    "threshold_mean",
                    "threshold_minimum",
                    "threshold_niblack",
                    "threshold_otsu",
                    "threshold_sauvola",
                    "threshold_triangle",
                    "threshold_yen",
                    "cv2.threshold",
                    "cv2.adaptiveThreshold",
                ]

                self.monochrome_method = random.choice(all_monochrome_method)
                print(self.monochrome_method)
                print("")

            # block size must present in the input argument
            if self.monochrome_method == "threshold_local":
                if self.monochrome_arguments is None:
                    self.monochrome_arguments = {}

                if "block_size" not in self.monochrome_arguments:
                    block_size = random.randint(31, min_size - 2)
                    # block size must be odd
                    if not block_size % 2:
                        block_size += 1
                    self.monochrome_arguments["block_size"] = block_size

            # window size of niblack and sauvola must be odd
            if (self.monochrome_method == "threshold_niblack") or (self.monochrome_method == "threshold_sauvola"):
                if self.monochrome_arguments and "window_size" in self.monochrome_arguments:
                    if not self.monochrome_arguments["window_size"] % 2:
                        self.monochrome_arguments["window_size"] += 1

            # cv2 threshold needs to have input arguments
            if self.monochrome_method == "cv2.threshold":
                if self.monochrome_arguments is None:
                    self.monochrome_arguments = {}
                if "thresh" not in self.monochrome_arguments:
                    self.monochrome_arguments["thresh"] = random.randint(64, 128)
                if "maxval" not in self.monochrome_arguments:
                    if image.dtype == "uint8":
                        maxval = 255
                    elif image.dtype == "float64":
                        maxval = 1.0
                    else:
                        maxval = np.max(image)
                    self.monochrome_arguments["maxval"] = maxval
                if "type" not in self.monochrome_arguments:
                    self.monochrome_arguments["type"] = cv2.THRESH_BINARY

            # cv2 adaptiveThreshold needs to have input arguments
            if self.monochrome_method == "cv2.adaptiveThreshold":
                if self.monochrome_arguments is None:
                    self.monochrome_arguments = {}
                if "maxValue" not in self.monochrome_arguments:
                    if image.dtype == "uint8":
                        maxValue = 255
                    elif image.dtype == "float64":
                        maxValue = 1.0
                    else:
                        maxValue = np.max(image)
                    self.monochrome_arguments["maxValue"] = maxValue

                if "adaptiveMethod" not in self.monochrome_arguments:
                    self.monochrome_arguments["adaptiveMethod"] = random.choice(
                        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C),
                    )

                if "thresholdType" not in self.monochrome_arguments:
                    self.monochrome_arguments["thresholdType"] = cv2.THRESH_BINARY

                if "blockSize" not in self.monochrome_arguments:
                    block_size = random.randint(31, min_size - 2)
                    if not block_size % 2:
                        block_size += 1
                    self.monochrome_arguments["blockSize"] = block_size

                if "C" not in self.monochrome_arguments:
                    self.monochrome_arguments["C"] = random.randint(1, 3)

        if self.halftone is None:
            self.halftone = random.choice((0, 0))
        if self.halftone:
            if self.invert is None:
                self.invert = 1
            if self.half_kernel_size is None:
                self.half_kernel_size = max(1, int(scale))
            if self.angle is None:
                self.angle = random.randint(0, 360)
            if self.sigma is None:
                self.sigma = random.randint(1, 3)

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run() or True:
            image = image.copy()
            self.init_default_values(image)

            # downscale image
            image_downscaled = self.downscale(image)

            # convert to gray
            image_out = self.rgb_to_gray(image_downscaled, invert=self.invert)

            # check and apply monochrome
            if self.monochrome:
                image_out = binary_threshold(
                    (image_out * 255).astype("uint8"),
                    self.monochrome_method,
                    self.monochrome_arguments,
                )

            # check and apply halftone
            if self.halftone:
                if self.monochrome:
                    image_out = image_out.astype("float") / 255

                image_out = self.generate_halftone(
                    image_out,
                    self.half_kernel_size,
                    self.angle,
                    self.sigma,
                )

                # check and invert image, then return image as uint8
                if self.invert:
                    image_out = ((1 - image_out) * 255).astype("uint8")
                else:
                    image_out = (image_out * 255).astype("uint8")

            # upscale image
            image_faxify = cv2.resize(image_out, (image.shape[1], image.shape[0]))

            return image_faxify

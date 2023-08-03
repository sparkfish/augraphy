"""
version: 0.0.1
*********************************

Dependencies:
- PIL
- opencv
- numpy

*********************************

References:

- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation: https://docs.opencv.org/4.x/

- Quasicrystals Inspiration: http://mainisusuallyafunction.blogspot.com/2011/10/quasicrystals-as-sums-of-waves-in-plane.html

*********************************



"""
import math
import random
import warnings

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class PatternGenerator(Augmentation):
    """In this implementation we take a geometric plane and every point in the plane is shaded according
    to its position,(x,y) coordinate. We take the pattern and perform a bitwise not operation so that it can
    be added as an background to an image.This code is a python implementation of a QuasiPattern Distortion augmentation techniques
    using PIL and the OpenCV libraries. This augmentation creates a new pattern image and superimposes it onto an input image.
    To make the pattern more prominent
    a. Increase the 'frequency' parameter: Increasing the frequency of the pattern will the it tightly populated and more prominent.
    b. Decrease the 'n_rotation' parameter: Decreasing the number of rotations will make the pattern less symmetrical.

    :param imgx: width of the pattern image. default is 512
    :type imgx: int, optional
    :param imgy: height of the pattern image, default is 512
    :type imgy: int, optional
    :param n_rotation: is the number of rotations applied to the pattern, default value lies
                       between 10 and 15.
    :type n_rotation: tuple (int) , optional
    :param color: Color of the pattern in BGR format. Use "random" for random color effect.
    :type color: tuple (int), optional
    :param alpha_range: Tuple of floats determining the alpha value of the patterns.
    :type alpha_range: tuple (float), optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
        :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        imgx=512,
        imgy=512,
        n_rotation_range=(10, 15),
        color="random",
        alpha_range=(0.25, 0.5),
        numba_jit=1,
        p=1.0,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.imgx = imgx  # width of the image
        self.imgy = imgy  # hieght of the image
        self.n_rotation_range = n_rotation_range  # number of rotation to be applied to the pattern
        self.color = color
        self.alpha_range = alpha_range
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def __repr__(self):
        # return f"QuasiPattern Distortion: width = {self.imgx} , height = {self.imgy}, n_rotation = {self.n_rotation}, color = {self.color}, alpha_range = {self.alpha_range}"
        return f"QuasiPattern Distortion(imgx={self.imgx}, imgy={self.imgy}, n_rotation_range = {self.n_rotation_range}, color={self.color}, alpha_range={self.alpha_range}, numba_jit={self.numba_jit}, p={self.p})"

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def apply_augmentation(ndim, pattern_image, frequency, phase, n_rotation):
        # Applies the Augmentation to input data.
        width, height = ndim
        # apply transformation, each pixel is transformed to cosine function
        for ky in range(height):
            y = np.float32(ky) / (height - 1) * 4 * math.pi - 2 * math.pi  # normalized coordinates of y-coordinate
            for kx in nb.prange(width):
                x = (
                    np.float32(kx) / (width - 1) * 4 * math.pi - 2 * math.pi
                )  # normalized coordinates of the x-coordinate
                z = 0.0  # z value will determine the intensity of the color, initially set to zero
                for i in nb.prange(n_rotation):
                    r = math.hypot(x, y)  # distance between the point to the origin
                    a = (
                        math.atan2(y, x) + i * math.pi * 2.0 / n_rotation
                    )  # angle the point makes to the origin plus rotation angle
                    z += math.cos(r * math.sin(a) * frequency + phase)  # function of cosine added as an offet
                c = int(round(255 * z / n_rotation))  # color
                pattern_image[ky, kx] = (c, c, c)  # RGB value

        return pattern_image

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w = result.shape[:2]

            self.n_rotation = random.randint(self.n_rotation_range[0], self.n_rotation_range[1])
            pattern_image = np.zeros((self.imgy, self.imgx, 3), dtype=np.uint8)
            frequency = random.random() * 100 + 18  # determines the frequency of pattern
            phase = random.random() * math.pi  # phase shift of the pattern
            ndim = (self.imgx, self.imgy)  # dimensions of pattern
            pattern = self.apply_augmentation(ndim, pattern_image, frequency, phase, n_rotation=self.n_rotation)
            invert = cv2.bitwise_not(pattern)  # performing bitwise not operation
            invert = cv2.resize(invert, (w, h), interpolation=cv2.INTER_LINEAR)
            if len(image.shape) < 3:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            sw = PatternMaker(alpha=random.uniform(self.alpha_range[0], self.alpha_range[1]))

            # apply color into pattern
            if self.color == "random":
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = self.color
            if len(invert.shape) > 2:
                color_mask = np.full_like(invert, fill_value=color, dtype="uint8")
            else:
                color_mask = np.full_like(invert, fill_value=np.mean(color), dtype="uint8")
            invert = cv2.multiply(invert, color_mask, scale=1 / 255)

            # overlay pattern into image
            result = sw.superimpose(result, invert)
            return result

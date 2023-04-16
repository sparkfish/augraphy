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

- PIL Documentation: https://pillow.readthedocs.io/en/stable/

- OpenCV Documentation: https://docs.opencv.org/4.x/

- Quasicrystals Inspiration: http://mainisusuallyafunction.blogspot.com/2011/10/quasicrystals-as-sums-of-waves-in-plane.html

*********************************



"""
import math
import os
import random
import warnings

import cv2
import numpy as np
from PIL import Image

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class PatternGenerator(Augmentation):
    def __init__(self, imgx=512, imgy=512, n_rotation=(10, 15), p=1.0):
        """
        In this implementation we take a geometric plane and every point in the plane is shaded according
        to its position,(x,y) coordinate. We take the pattern and perform a bitwise not operation so that it can
        be added as an background to an image.This code is a python implementation of a QuasiPattern Distortion augmentation techniques
        using PIL and the OpenCV libraries. This augmentation creates a new pattern image and superimposes it onto an input image.

        To make the pattern more prominent
        a. Increase the 'frequency' parameter: Increasing the frequency of the pattern will the it tightly populated and more prominent.
        b. Decrease the 'n_rotation' parameter: Decreasing the number of rotations will make the pattern less symmetrical.

        :param imgx: width of the pattern image. default is 512
        :type imgx: int
        :param imgy: height of the pattern image, default is 512
        :type imgy: int
        :param n_rotation: is the number of rotations applied to the pattern, default value lies
                           between 10 and 15.
        :type n_rotation: tuple (int)


        """
        super().__init__(p=p)
        self.imgx = imgx  # width of the image
        self.imgy = imgy  # hieght of the image
        self.n_rotation = random.randint(
            n_rotation[0],
            n_rotation[1],
        )  # number of rotation to be applied to the pattern

    def __repr__(self):
        return f"QuasiPattern Distortion: width = {self.imgx} , height = {self.imgy}, n_rotation = {self.n_rotation}"

    def apply_augmentation(self):
        # Applies the Augmentation to input data.
        pattern_image = Image.new("RGB", (self.imgx, self.imgy))
        pixels = pattern_image.load()
        frequency = random.random() * 100 + 18  # determines the frequency of pattern
        phase = random.random() * math.pi  # phase shift of the pattern
        # apply transformation, each pixel is transformed to cosine function
        for ky in range(self.imgy):
            y = float(ky) / (self.imgy - 1) * 4 * math.pi - 2 * math.pi  # normalized coordinates of y-coordinate
            for kx in range(self.imgx):
                x = (
                    float(kx) / (self.imgx - 1) * 4 * math.pi - 2 * math.pi
                )  # normalized coordinates of the x-coordinate
                z = 0.0  # z value will determine the intensity of the color, initially set to zero
                for i in range(self.n_rotation):
                    r = math.hypot(x, y)  # distance between the point to the origin
                    a = (
                        math.atan2(y, x) + i * math.pi * 2.0 / self.n_rotation
                    )  # angle the point makes to the origin plus rotation angle
                    z += math.cos(r * math.sin(a) * frequency + phase)  # function of cosine added as an offet
                c = int(round(255 * z / self.n_rotation))  # color
                pixels[kx, ky] = (c, c, c)  # # RGB value
        # pattern_image = ImageOps.autocontrast(pattern_image) # increasing the contrast of the image
        pattern_image.save(
            "images/guilloche_pattern.png",
            "PNG",
            dpi=(300, 300),
        )  # save the image as DPI = 300 to make it standard
        pattern = cv2.imread("images/guilloche_pattern.png")  # load the image
        os.remove("images/guilloche_pattern.png")  # delete the image

        pattern = np.array(pattern_image)  # returning numpy array
        return pattern

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w, _ = result.shape
            pattern = self.apply_augmentation()
            invert = cv2.bitwise_not(pattern)  # performing bitwise not operation
            invert = cv2.resize(invert, (w, h), interpolation=cv2.INTER_LINEAR)
            sw = PatternMaker(alpha=0.45)
            result = sw.superimpose(result, invert)
            return result

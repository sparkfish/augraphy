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
import os
import random
import warnings

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class PatternGenerator(Augmentation):
    def __init__(self, imgx=512, imgy=512, n_rotation_range=(10, 15), p=1.0):
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
        self.n_rotation_range = n_rotation_range  # number of rotation to be applied to the pattern

    def __repr__(self):
        # return f"QuasiPattern Distortion: width = {self.imgx} , height = {self.imgy}, n_rotation = {self.n_rotation}"
        return f"QuasiPattern Distortion:  n_rotation_range = {self.n_rotation_range}"

    def apply_augmentation(self):
        # Applies the Augmentation to input data.
        pattern_image = np.zeros((self.imgy, self.imgx, 3), dtype=np.uint8)
        frequency = random.random() * 100 + 18  # determines the frequency of pattern
        phase = random.random() * math.pi  # phase shift of the pattern
        # apply transformation, each pixel is transformed to cosine function
        for ky in range(self.imgy):
            y = np.float32(ky) / (self.imgy - 1) * 4 * math.pi - 2 * math.pi  # normalized coordinates of y-coordinate
            for kx in range(self.imgx):
                x = (
                    np.float32(kx) / (self.imgx - 1) * 4 * math.pi - 2 * math.pi
                )  # normalized coordinates of the x-coordinate
                z = 0.0  # z value will determine the intensity of the color, initially set to zero
                for i in range(self.n_rotation):
                    r = math.hypot(x, y)  # distance between the point to the origin
                    a = (
                        math.atan2(y, x) + i * math.pi * 2.0 / self.n_rotation
                    )  # angle the point makes to the origin plus rotation angle
                    z += math.cos(r * math.sin(a) * frequency + phase)  # function of cosine added as an offet
                c = int(round(255 * z / self.n_rotation))  # color
                pattern_image[ky, kx] = (c, c, c)  # RGB value

        return pattern_image

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w = result.shape[:2]
            self.n_rotation = random.randint(self.n_rotation_range[0], self.n_rotation_range[1])
            pattern = self.apply_augmentation()
            invert = cv2.bitwise_not(pattern)  # performing bitwise not operation
            invert = cv2.resize(invert, (w, h), interpolation=cv2.INTER_LINEAR)
            if len(image.shape) < 3:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            sw = PatternMaker(alpha=0.15)
            result = sw.superimpose(result, invert)
            return result

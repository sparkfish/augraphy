'''
version: 0.0.1
*********************************

Dependencies: 
- PIL
- opencv

*********************************

References: 

- PIL Documentation: https://pillow.readthedocs.io/en/stable/

- OpenCV Documentation: https://docs.opencv.org/4.x/

- Quasicrystals Inspiration: http://mainisusuallyafunction.blogspot.com/2011/10/quasicrystals-as-sums-of-waves-in-plane.html

*********************************



'''
import random
import math
from PIL import Image
import cv2
import os
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.slidingwindow import PatternMaker

class PatternGenerator(Augmentation):
    def __init__(self, imgx = 512, imgy = 512, n_rotation = (30,40), p= 1.0):
        '''
        This code is a python implementation of a QuasiPattern Distortion augmentation techniques using PIL and the 
        OpenCV libraries. This augmentation creates a new pattern image and superimposes it onto an input image.
        
        :param imgx: width of the pattern image. default is 512
        :type imgx: int
        :param imgy: height of the pattern image, default is 512
        :type imgy: int
        :param n_rotation: is the number of rotations applied to the pattern, default value lies
                           between 30 and 40.
        :type n_rotation: tuple (int)


        '''
        super().__init__(p=p)
        self.imgx = imgx
        self.imgy = imgy
        self.n_rotation = random.randint(n_rotation[0], n_rotation[1])

    def __repr__(self):
        return f"QuasiPattern Distortion: width = {self.imgx} , height = {self.imgy}, n_rotation = {self.n_rotation}"
    
    def apply_augmentation(self):
        # Applies the Augmentation to input data.
        pattern_image = Image.new("RGB", (self.imgx, self.imgy))
        pixels = pattern_image.load()
        frequency = random.random() * 100 + 10
        phase = random.random() * math.pi

        for ky in range(self.imgy):
            y = float(ky) / (self.imgy - 1) * 4 * math.pi - 2 * math.pi
            for kx in range(self.imgx):
                x = float(kx) / (self.imgx - 1) * 4 * math.pi - 2 * math.pi
                z = 0.0
                for i in range(self.n_rotation):
                    r = math.hypot(x, y)
                    a = math.atan2(y, x) + i * math.pi * 3.0 / self.n_rotation
                    z += math.cos(r * math.sin(a) * frequency + phase)
                c = int(round(255 * z / self.n_rotation))
                pixels[kx, ky] = (c, c, c) # grayscale
        pattern_image.save("images/guilloche_pattern.png", "PNG")
        pattern = cv2.imread("images/guilloche_pattern.png")
        os.remove("images/guilloche_pattern.png")
        return pattern
        
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w, _ = result.shape
            pattern = self.apply_augmentation()
            invert = cv2.bitwise_not(pattern) # performing bitwise not operation
            invert = cv2.resize(invert, (w, h), interpolation= cv2.INTER_LINEAR)
            sw = PatternMaker(alpha = 0.48) 
            result = sw.superimpose(result, invert)
            return result
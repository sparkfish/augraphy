import numpy as np
import random
import cv2
import math
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class bookbinding(Augmentation):
    # creates a book binding effect with shadow and curved lines

    def __init__(self,radius_range=(1,100),
                 angle_range=(20,75),
                 curve_intensity_range=(0,70),
                 p=0.5):
        super.__init__(p=p)
        self.radius_range=radius_range
        self.angle_range=angle_range
        self.curve_intensity_range=curve_intensity_range

    def __repr__(self):
        return f"bookbinding(radius_range={self.radius_range}, angle_range={self.angle_range}, curve_intensity_range={self.curve_intensity_range},  p={self.p})"

    def add_book_shadow(self,img,radius, angle):
        img_output = img.copy()

        cols=img.shape[0]
        rows=img.shape[1]
        for i in range(rows):
            for j in range(cols):
                dist = math.sqrt((j * j))
                d = dist + (radius * (1 - math.cos(angle)))
                new_i = img[i, j] * (dist / (d)) ** 2
                img_output[i, j, :] = new_i
        return img_output
    def __call__(self,  data,force=False):
        radius = random.randint(self.radius_range[0],self.radius_range[1])
        angle=random.randint(self.angle_range[0],self.angle_range[1])
        curve_intensity =random.randint(self.curve_intensity_range[0],self.curve_intensity_range[1])
        image = data["post"][-1].result.copy()
        image=self.add_book_shadow(image,radius,angle)

        data["post"].append(AugmentationResult(self, image))
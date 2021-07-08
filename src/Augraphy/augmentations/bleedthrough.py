################################################################################
# File: bleedthrough.py
#
# Class: BleedThroughAugmentation
#
# Description: This file contains a class defining an Augmentation which uses 
#              combination of inkbleed and gaussian blur augmentation.
#              
################################################################################


################################################################################
# Imports
################################################################################

import numpy as np
import random
import cv2
from Augraphy.Augmentations import *

################################################################################
# Class Definition
################################################################################


class BleedThroughAugmentation(Augmentation):
    
    def __init__(
        self,
        intensity_range=(0.1, 0.2), 
        color_range=(0, 224), 
        ksize = (17,17),
        sigmaX = 0,
        probability=0.5,
    ):
        super().__init__(probability=probability)
        self.intensity_range = intensity_range
        self.color_range = color_range
        self.ksize = ksize
        self.sigmaX  = sigmaX 


    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BleedThroughAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, ksize={self.ksize}, sigmaX={self.sigmaX},probability={self.probability})"


    # Add salt and pepper noise
    def add_sp_noise(self, img, prob=0.05):
        output = np.zeros_like(img)
        thres = 1 - prob 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 255
                elif rdn > thres:
                    output[i][j] = 0
                else:
                    output[i][j] = img[i][j]
        return output


    # Computes the gradient of the image intensity function.
    def sobel(self, image):
        gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient


    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data["ink"][-1].result.copy()
            # img_flip = cv2.flip(img, 1) # flip is not working for rotated image
            img_flip = img
            
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            add_noise_fn = (
                lambda x, y: random.randint(self.color_range[0], self.color_range[1])
                if (y == 255 and random.random() < intensity)
                else x
            )
            add_noise = np.vectorize(add_noise_fn)
            sobel = self.sobel(img_flip)
            img_noise = add_noise(img_flip, self.add_sp_noise(sobel))
            img_blur = cv2.GaussianBlur(img_noise, ksize=self.ksize, sigmaX=self.sigmaX)  

            # internal parameters, can be improved later
            x_offset = 40;
            y_offset = 40;
            bleed_through_intensity = 1

            # shift the blurred words
            img_new = img.copy()
            if len(img_new.shape)>2:
                img_new[y_offset:,x_offset:,:]  = np.uint8( ((img[y_offset:,x_offset:,:].astype('float')) + ((img_blur[:-y_offset,:-x_offset,:].astype('float')/(1/1)))   )  /2) # 1/1 is reserved for intensity change
                img_new[:y_offset,:,:]  = np.uint8( ((img[:y_offset,:,:].astype('float')) + ((img_blur[:y_offset,:].astype('float')/(1/bleed_through_intensity)))   )  /2)
                img_new[:,:x_offset,:]  = np.uint8( ((img[:,:x_offset,:].astype('float')) + ((img_blur[:,:x_offset].astype('float')/(1/bleed_through_intensity)))   )  /2)
            else:
                img_new[y_offset:,x_offset:]  = np.uint8( ((img[y_offset:,x_offset:].astype('float')) + ((img_blur[:-y_offset,:-x_offset].astype('float')/(1/1)))   )  /2)
                img_new[:y_offset,:]  = np.uint8( ((img[:y_offset,:].astype('float')) + ((img_blur[:y_offset,:].astype('float')/(1/bleed_through_intensity)))   )  /2)
                img_new[:,:x_offset]  = np.uint8( ((img[:,:x_offset].astype('float')) + ((img_blur[:,:x_offset].astype('float')/(1/bleed_through_intensity)))   )  /2)
                


            data["ink"].append(AugmentationResult(self, img_new))

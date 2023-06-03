import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class ColorShift(Augmentation):
    """Shifts each BGR color channel by certain offsets to create a shifted color effect.

    :param color_shift_offset_x_range: Pair of ints/floats determining the value of x offset in shifting each color channel.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the x offset will be scaled by image width: 
            x offset (int) = image width  * x offset (float and 0.0 - 1.0)
    :type color_shift_offset_x_range: tuple, optional   
    :param color_shift_offset_y_range: Pair of ints/floats determining the value of y offset in shifting each color channel.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the y offset will be scaled by image height: 
            y offset (int) = image height  * y offset (float and 0.0 - 1.0)
    :type color_shift_offset_y_range: tuple, optional
    :param color_shift_iterations: Pair of ints determining the number of iterations in applying the color shift operation.
    :type color_shift_iterations: tuple, optional
    :param color_shift_brightness_range: Pair of floats determining the brightness value of the shifted color channel.
            The optimal brightness range is 0.9 to 1.1.
    :type color_shift_brightness_range: tuple, optional
    :param color_shift_gaussian_kernel_range : Pair of floats determining the brightness value of the shifted color channel.
    :type color_shift_gaussian_kernel_range : tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        color_shift_offset_x_range = (3,5),
        color_shift_offset_y_range = (3,5),
        color_shift_iterations = (1,2),
        color_shift_brightness_range = (0.9,1.1),
        color_shift_gaussian_kernel_range = (3,3),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.color_shift_offset_x_range = color_shift_offset_x_range
        self.color_shift_offset_y_range = color_shift_offset_y_range
        self.color_shift_iterations = color_shift_iterations
        self.color_shift_brightness_range = color_shift_brightness_range
        self.color_shift_gaussian_kernel_range = color_shift_gaussian_kernel_range

    def __repr__(self):
        return f"ColorShift(color_shift_offset_x_range={self.color_shift_offset_x_range}, color_shift_offset_y_range={self.color_shift_offset_y_range}, color_shift_iterations={self.color_shift_iterations}, color_shift_brightness_range={self.color_shift_brightness_range}, color_shift_gaussian_kernel_range={self.color_shift_gaussian_kernel_range}, p={self.p})"

    def apply_color_shift(self, image):
        """ Main function to apply color shift process.
        
        :param image: The input image.
        :type image: numpy array
        """
        
        ysize, xsize = image.shape[:2]
            
        image_b, image_g, image_r = cv2.split(image.copy())
        images = [image_b, image_g, image_r]
            
        brightness_ratio = random.uniform(self.color_shift_brightness_range[0], self.color_shift_brightness_range[1])
        
        # possible combinations in term of x and y direction
        directions = [[1,1], [1,-1], [-1,1], [-1,-1]]
        for i, image_single_color in enumerate(images):
            
            # get random direction
            index = random.randint(0, len(directions)-1)
            direction_x, direction_y = directions.pop(index)
            
            # generate random offsets
            if self.color_shift_offset_x_range[0] <=1.0 and isinstance(self.color_shift_offset_x_range[0], float):
                offset_x = random.randint(int(self.color_shift_offset_x_range[0]* xsize),int(self.color_shift_offset_x_range[1]* xsize))
            else:
                offset_x = random.randint(self.color_shift_offset_x_range[0],  self.color_shift_offset_x_range[1])
            if self.color_shift_offset_y_range[0] <=1.0 and isinstance(self.color_shift_offset_y_range[0], float):
                offset_y = random.randint(int(self.color_shift_offset_y_range[0]* ysize),int(self.color_shift_offset_y_range[1]* ysize))
            else:
                offset_y = random.randint(self.color_shift_offset_y_range[0],  self.color_shift_offset_y_range[1])

            # y direction
            translation_matrix_y = np.float32([[1, 0, 0], [0, 1, offset_y* direction_y]])
            # get a copy of translated area
            if direction_y>0:
                image_patch = image_single_color[-offset_y:, :].copy()
            else:   
                image_patch = image_single_color[:offset_y, :].copy()
            # shift image in y direction
            image_single_color = cv2.warpAffine(image_single_color, translation_matrix_y, (xsize, ysize))
            # fill back the empty area after translation
            if direction_y>0:
                image_single_color[:offset_y, :] = image_patch
            else:   
                image_single_color[-offset_y:, :] = image_patch
            
            # x direction
            translation_matrix_x = np.float32([[1, 0, offset_x * direction_x], [0, 1, 0]])
            # get a copy of translated area
            if direction_x>0:
                image_patch = image_single_color[:, -offset_x:].copy()
            else:   
                image_patch = image_single_color[:, :offset_x].copy()
            # shift image in x direction
            image_single_color = cv2.warpAffine(image_single_color, translation_matrix_x, (xsize, ysize))
            # fill back the empty area after translation
            if direction_x>0:
                image_single_color[:, :offset_x] = image_patch
            else:   
                image_single_color[:, -offset_x:] = image_patch

            # apply random brightness
            image_single_color_ratio = image_single_color.astype("int")* brightness_ratio
            image_single_color_ratio[image_single_color_ratio>255] = 255
            image_single_color_ratio[image_single_color_ratio<0] = 0
            image_single_color_ratio = image_single_color_ratio.astype("uint8")
            
            # blur the shhifted image
            kernel_value = random.randint(self.color_shift_gaussian_kernel_range[0], self.color_shift_gaussian_kernel_range[1])
            # kernel must be odd
            if not (kernel_value % 2):
                kernel_value += 1
            image_single_color_ratio = cv2.GaussianBlur(image_single_color_ratio, (kernel_value,kernel_value),0)
            
            # reassign the shifted color channel back to the image
            image[:,:,i] = image_single_color_ratio

        return image
        
        
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            
            
            image_output = image.copy()
  
            # convert and make sure image is color image
            if len(image_output.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image_output = cv2.cvtColor(image_output, cv2.COLOR_GRAY2BGR)
    
            # apply color shift based on the generated iterations
            color_shift_iterations = random.randint(self.color_shift_iterations[0], self.color_shift_iterations[1])
            for i in range(color_shift_iterations):
                image_output = self.apply_color_shift(image_output)
                
            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
                
            return image_output
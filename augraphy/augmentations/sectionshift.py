import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class SectionShift(Augmentation):
    """Shift single or multiple sections of image in horizontal, vertical or both directions to create an effect of shifted image sections.

    :param section_shift_number_range: Tuple of ints determing the number of section shift operation.
    :type section_shift_number_range: tuple, optional
    :param section_shift_locations: A nested list contains list of shifting boxes. 
        Each box should be in format of [x0, y0, xn, yn].
        Use "random" for random location.
    :type section_shift_locations: list, optional
    :param section_shift_x_range: Tuple of ints determing the shifting value in horizontal direction.
        The shifting value will be in percentage of the image width if the value is float and in between -1.0 - 1.0:
        shifting_x (int) = image width  * shifting_x (float and -1.0 - 1.0)
    :type section_shift_x_range: tuple, optional
    :param section_shift_y_range: Tuple of ints determing the shifting value in vertical direction.
        The shifting value will be in percentage of the image height if the value is float and in between -1.0 - 1.0:
        shifting_y (int) = image height  * shifting_y (float and -1.0 - 1.0)
    :type section_shift_y_range: tuple, optional
    :param section_shift_fill_value: Tuple of values in BGR to fill in the shifted area.
        Use "-1" to not fill any value and the image default value will be used instead.
        Use "random" to fill random color.
    :type section_shift_fill_value: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        section_shift_number_range = (3,5),
        section_shift_locations = "random",
        section_shift_x_range = (-10,10),
        section_shift_y_range = (-10,10),
        section_shift_fill_value = -1,        
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.section_shift_number_range = section_shift_number_range
        self.section_shift_locations = section_shift_locations
        self.section_shift_x_range = section_shift_x_range
        self.section_shift_y_range = section_shift_y_range
        self.section_shift_fill_value = section_shift_fill_value

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SectionShift(section_shift_number_range={self.section_shift_number_range}, section_shift_locations={self.section_shift_locations}, section_shift_x_range={self.section_shift_x_range}, section_shift_y_range={self.section_shift_y_range},  section_shift_fill_value={self.section_shift_fill_value}, p={self.p})"

    def apply_shift(self, image, shift_box, section_shift_x, section_shift_y):
        """Core function to shift section of image based on the input box and shifting values.

        :param image: The input image.
        :type image: numpy array
        :param shift_box: Tuple contains the box of the shifting location in format of x0, y0, xn, yn.
        :type shift_box: tuple
        :param section_shift_x: The shifting value in horizontal direction.
        :type section_shift_x: int
        :param section_shift_y: The shifting value in vertical direction.
        :type section_shift_y: int
        """

        ysize, xsize = image.shape[:2]
        x0,y0,xn,yn = shift_box

        # make sure doesn't exceed image boundary
        x0 = min(xsize-section_shift_x-1, x0)
        y0 = min(ysize-section_shift_y-1, y0)
        if x0 + section_shift_x<0:
            x0 = x0 - section_shift_x
        if y0 + section_shift_y<0:
            y0 = y0 - section_shift_y
        if xn + section_shift_x>xsize:
            xn = xsize - section_shift_x
        if yn + section_shift_y>ysize:
            yn = ysize - section_shift_y
            
        # the section of shifted image
        image_section = image[y0:yn, x0:xn].copy()
    
        # fill the shifted area with value
        if self.section_shift_fill_value != -1:
            if self.section_shift_fill_value == "random":
                image[y0:yn, x0:xn] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
            else:
                image[y0:yn, x0:xn] = self.section_shift_fill_value
    
        # shift the section of image
        image[y0+section_shift_y:yn+section_shift_y, x0+section_shift_x:xn+section_shift_x] = image_section
        

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            ysize, xsize = image.shape[:2]
            
            # generate number of shifting operation
            if self.section_shift_locations == "random":
                section_shift_number = random.randint(self.section_shift_number_range[0], self.section_shift_number_range[1])
            else:
                section_shift_number = len(self.section_shift_locations)
            
            for i in range(section_shift_number):
                
                # check input to scale it with image width
                if isinstance(self.section_shift_x_range[1], float) and self.section_shift_x_range[1] <= 1 and self.section_shift_x_range[1] >= -1:
                    section_shift_x = random.randint(int(self.section_shift_x_range[0] * xsize), int(self.section_shift_x_range[1] * xsize))
                else:
                    section_shift_x = random.randint(self.section_shift_x_range[0], self.section_shift_x_range[1])
                    
                # check input to scale it with image height
                if isinstance(self.section_shift_y_range[1], float) and self.section_shift_y_range[1] <= 1 and self.section_shift_y_range[1] >= -1:
                    section_shift_y = random.randint(int(self.section_shift_y_range[0] * ysize), int(self.section_shift_y_range[1] * ysize))
                else: 
                    section_shift_y = random.randint(self.section_shift_y_range[0], self.section_shift_y_range[1])
                
                if self.section_shift_locations == "random":    
                    # for random section, generate random section width and height
                    section_shift_width_size = random.randint(int(xsize/20), int(xsize/5))
                    section_shift_height_size = random.randint(int(ysize/20), int(ysize/5))
 
                    # generate random box
                    start_x = random.randint(0, xsize-section_shift_x-section_shift_width_size-1)
                    start_y = random.randint(0, ysize-section_shift_y-section_shift_height_size-1)       
                    end_x = start_x + section_shift_width_size
                    end_y = start_y + section_shift_height_size 
                    shift_box = [start_x, start_y, end_x, end_y]
                else:
                    shift_box = self.section_shift_locations[i]
                     
                # apply section shift
                self.apply_shift(image, shift_box, section_shift_x, section_shift_y)
                
            # return image follows the input image color channel
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image

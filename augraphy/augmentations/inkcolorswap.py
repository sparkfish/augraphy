import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class InkColorSwap(Augmentation):
    """Swap color of ink in the image based on detected ink contours.

    :param ink_swap_color: The swapping color (in BGR) of the effect.
    :type ink_swap_color: tuple, optional
    :param ink_swap_sequence_number_range: Pair of ints determing the consecutive swapping number in the detected contours.
    :type ink_swap_sequence_number_range: tuple, optional
    :param ink_swap_min_width_range: Pair of ints/floats determining the minimum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum width will be scaled by image width:
            min width (int) = image width  * min width (float and 0.0 - 1.0)
    :type ink_swap_min_width_range: tuple, optional
    :param ink_swap_max_width_range: Pair of ints/floats determining the maximum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum width will be scaled by image width:
            max width (int) = image width  * max width (float and 0.0 - 1.0)
    :type ink_swap_max_width_range: tuple, optional
    :param ink_swap_min_height_range: Pair of ints/floats determining the minimum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum height will be scaled by image height:
            min height (int) = image height  * min height (float and 0.0 - 1.0)
    :type ink_swap_min_height_range: tuple, optional
    :param ink_swap_max_height_range: Pair of ints/floats determining the maximum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum height will be scaled by image height:
            max height (int) = image height  * max height (float and 0.0 - 1.0)
    :type ink_swap_max_height_range: tuple, optional
    :param ink_swap_min_area_range: Pair of ints/floats determining the minimum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum area will be scaled by image area:
            min area (int) = image area  * min area (float and 0.0 - 1.0)
    :type ink_swap_min_area_range: tuple, optional
    :param ink_swap_max_area_range: Pair of ints/floats determining the maximum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum area will be scaled by image area:
            max area (int) = image area  * max area (float and 0.0 - 1.0)
    :type ink_swap_max_area_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        ink_swap_color = "random",
        ink_swap_sequence_number_range = (5,10),
        ink_swap_min_width_range=(2,3),
        ink_swap_max_width_range=(100,120),
        ink_swap_min_height_range=(2,3),
        ink_swap_max_height_range=(100,120),
        ink_swap_min_area_range=(10,20),
        ink_swap_max_area_range=(400,500),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.ink_swap_color = ink_swap_color
        self.ink_swap_sequence_number_range = ink_swap_sequence_number_range
        self.ink_swap_min_width_range = ink_swap_min_width_range
        self.ink_swap_max_width_range = ink_swap_max_width_range
        self.ink_swap_min_height_range = ink_swap_min_height_range
        self.ink_swap_max_height_range = ink_swap_max_height_range
        self.ink_swap_min_area_range = ink_swap_min_area_range
        self.ink_swap_max_area_range = ink_swap_max_area_range
        
    def __repr__(self):
        return f"InkColorSwap(ink_swap_color={self.ink_swap_color}, ink_swap_sequence_number_range={self.ink_swap_sequence_number_range}, ink_swap_min_width_range={self.ink_swap_min_width_range}, ink_swap_max_width_range={self.ink_swap_max_width_range}, ink_swap_min_height_range={self.ink_swap_min_height_range}, ink_swap_max_height_range={self.ink_swap_max_height_range}, ink_swap_min_area_range={self.ink_swap_min_area_range}, ink_swap_max_area_range={self.ink_swap_max_area_range}, p={self.p})"


    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            image = image.copy()
            
            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # get iamge size and area
            ysize, xsize = image.shape[:2]
            image_area = ysize * xsize

            # generate mask
            image_mask = np.zeros((ysize, xsize),dtype="uint8")

            # width
            if self.ink_swap_min_width_range[0] <= 1.0 and isinstance(self.ink_swap_min_width_range[0], float):
                min_width = random.randint(int(self.ink_swap_min_width_range[0]*xsize), int(self.ink_swap_min_width_range[1]*xsize))
            else:
                min_width = random.randint(self.ink_swap_min_width_range[0], self.ink_swap_min_width_range[1])
            if self.ink_swap_max_width_range[0] <= 1.0 and isinstance(self.ink_swap_max_width_range[0], float):
                max_width = random.randint(int(self.ink_swap_max_width_range[0]*xsize), int(self.ink_swap_max_width_range[1]*xsize))
            else: 
                max_width = random.randint(self.ink_swap_max_width_range[0], self.ink_swap_max_width_range[1])

            # height
            if self.ink_swap_min_height_range[0] <= 1.0 and isinstance(self.ink_swap_min_height_range[0], float):
                min_height = random.randint(int(self.ink_swap_min_height_range[0]*ysize), int(self.ink_swap_min_height_range[1]*ysize))
            else:
                min_height = random.randint(self.ink_swap_min_height_range[0], self.ink_swap_min_width_range[1])
            if self.ink_swap_max_height_range[0] <= 1.0 and isinstance(self.ink_swap_max_height_range[0], float):
                max_height = random.randint(int(self.ink_swap_max_height_range[0]*ysize), int(self.ink_swap_max_height_range[1]*ysize))
            else: 
                max_height = random.randint(self.ink_swap_max_height_range[0], self.ink_swap_max_height_range[1])

            # area
            if self.ink_swap_min_area_range[0] <= 1.0 and isinstance(self.ink_swap_min_area_range[0], float):
                min_area = random.randint(int(self.ink_swap_min_area_range[0]*image_area), int(self.ink_swap_min_area_range[1]*image_area))
            else:
                min_area = random.randint(self.ink_swap_min_area_range[0], self.ink_swap_min_area_range[1])
            if self.ink_swap_max_area_range[0] <= 1.0 and isinstance(self.ink_swap_max_area_range[0], float):
                max_area = random.randint(int(self.ink_swap_max_area_range[0]*image_area), int(self.ink_swap_max_area_range[1]*image_area))
            else: 
                max_area = random.randint(self.ink_swap_max_area_range[0], self.ink_swap_max_area_range[1])

            # convert input image to gray
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #  convert image into binary
            _, image_binary = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # find contours of image
            contours, _ = cv2.findContours(
                    image_binary,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
    
            color_mode = 1
            ink_swap_sequence_number = random.randint(self.ink_swap_sequence_number_range[0], self.ink_swap_sequence_number_range[1])
    
            # draw mask
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if w<max_width and w>min_width and h<max_height and h>min_height and area<max_area and area>min_area:
    
                    # draw contour for swap color
                    if color_mode:
                        cv2.drawContours(image_mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)
            
                    # reduce count for contour, and change color when count <= 0
                    ink_swap_sequence_number -= 1
                    if ink_swap_sequence_number <= 0:
                        ink_swap_sequence_number = random.randint(self.ink_swap_sequence_number_range[0], self.ink_swap_sequence_number_range[1])
                        color_mode = 1- color_mode

            if self.ink_swap_color == "random":
                ink_swap_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                ink_swap_color = self.ink_swap_color

            # create a mask of swap color
            image_color = np.full_like(image, fill_value=ink_swap_color, dtype="uint8")
            
            # blend image with swap color
            image_color = cv2.addWeighted(image, 1.0, image_color, 1.0, 0)
            
            # update image to blended image in the contour area
            image[image_mask>0] = image_color[image_mask>0]

            # return image follows the input image color channel
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image

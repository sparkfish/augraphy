import random

import cv2
import numpy as np
from PIL import Image

from augraphy.base.augmentation import Augmentation


class DotMatrix(Augmentation):
    """Creates a dot matrix image by filling image with dots of mean color in different shapes.

    :param dot_matrix_shape: The shape of single dot in dot matrix effect.
        The existing shapes are "cicle", "rectangle", "triangle" and "diamond".
        Use "random" to select shape randomly.
    :type dot_matrix_shape: string, optional
    :param dot_matrix_dot_width_range: Tuple of ints determining the width of single dot in dot matrix effect.
    :type dot_matrix_dot_width_range: tuple, optional
    :param dot_matrix_dot_height_range: Tuple of ints determining the height of single dot in dot matrix effect.
    :type dot_matrix_dot_height_range: tuple, optional
    :param dot_matrix_gaussian_kernel_value_range: Tuple of ints determining the Gaussian kernel value in blurring the dot matrix image.
    :type dot_matrix_gaussian_kernel_value_range: tuple, optional
    :param dot_matrix_rotate_value_range: Tuple of ints determining the angle of rotation of the dot matrix effect.
    :type dot_matrix_rotate_value_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        dot_matrix_shape = "random",
        dot_matrix_dot_width_range = (13, 19),
        dot_matrix_dot_height_range = (13, 19),
        dot_matrix_gaussian_kernel_value_range = (9,15),
        dot_matrix_rotate_value_range = (0,360),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.dot_matrix_shape = dot_matrix_shape
        self.dot_matrix_dot_width_range = dot_matrix_dot_width_range
        self.dot_matrix_dot_height_range = dot_matrix_dot_height_range
        self.dot_matrix_gaussian_kernel_value_range = dot_matrix_gaussian_kernel_value_range
        self.dot_matrix_rotate_value_range = dot_matrix_rotate_value_range

    def __repr__(self):
        return f"DotMatrix(dot_matrix_shape={self.dot_matrix_shape}, dot_matrix_dot_width_range={self.dot_matrix_dot_width_range}, dot_matrix_dot_height_range={self.dot_matrix_dot_width_range}, dot_matrix_gaussian_kernel_value_range={self.dot_matrix_gaussian_kernel_value_range}, dot_matrix_rotate_value_range={self.dot_matrix_rotate_value_range}, p={self.p})"

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            ysize, xsize = image.shape[:2]

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # generate shape
            if self.dot_matrix_shape == "random":
                dot_matrix_shape = random.choice(["circle", "rectangle", "triangle", "diamond"])
            else:
                dot_matrix_shape = self.dot_matrix_shape

            # generate each dot height and width
            dot_matrix_dot_width = random.randint(self.dot_matrix_dot_width_range[0], self.dot_matrix_dot_width_range[1])
            dot_matrix_dot_height = random.randint(self.dot_matrix_dot_height_range[0], self.dot_matrix_dot_height_range[1])

            # each dot width or height must be odd
            if not dot_matrix_dot_width % 2:
                dot_matrix_dot_width += 1
            if not dot_matrix_dot_height % 2:
                dot_matrix_dot_height += 1  
            
            if dot_matrix_shape == "circle":
                # min size of dot is 3 pixels for circle
                dot_matrix_dot_width = max(3, dot_matrix_dot_width)
                dot_matrix_dot_height = max(3, dot_matrix_dot_height)
                # initialize dot image
                image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                # draw shape
                center_x = int(np.floor(dot_matrix_dot_width/2))
                center_y = int(np.floor(dot_matrix_dot_height/2))
                radius = int(np.floor(min(dot_matrix_dot_width/2, dot_matrix_dot_height/2)))
                cv2.circle(image_dot, (center_x, center_y), radius, (255,255,255), -1)

            elif dot_matrix_shape == "rectangle":   
                # min size of dot is 3 pixels for rectangle
                dot_matrix_dot_width = max(3, dot_matrix_dot_width)
                dot_matrix_dot_height = max(3, dot_matrix_dot_height)
                # initialize dot image
                image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                # draw shape
                image_dot[1:-1, 1:-1] = 255

            elif dot_matrix_shape == "triangle":
                # min size of dot is 5 pixels for triangle
                dot_matrix_dot_width = max(5, dot_matrix_dot_width)
                dot_matrix_dot_height = max(5, dot_matrix_dot_height)
                # initialize dot image
                image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                # draw shape (index starts from 1 to leave 1 space for black pixel)
                y0 = 1
                yn = dot_matrix_dot_height-2
                x0 = 1
                xmid = int(np.floor(dot_matrix_dot_width/2))
                xn = dot_matrix_dot_width -2
                triangle_points = np.array([(x0,yn), (xmid, y0), (xn,yn)])
                cv2.drawContours(image_dot, [triangle_points], 0, (255,255,255), -1)       
                # mirror left right for consistent shape
                image_dot[:, :xmid] = np.fliplr(image_dot[:, xmid+1:])

            elif dot_matrix_shape == "diamond":
                # min size of dot is 5 pixels for triangle
                dot_matrix_dot_width = max(5, dot_matrix_dot_width)
                dot_matrix_dot_height = max(5, dot_matrix_dot_height)
                # initialize dot image
                image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                # draw shape
                y0 = 1
                ymid = int(np.floor(dot_matrix_dot_height/2))
                yn = dot_matrix_dot_height-2
                x0 = 1
                xmid = int(np.floor(dot_matrix_dot_width/2))
                xn = dot_matrix_dot_width-2
                triangle_points = np.array([(x0,ymid), (xmid, y0), (xn, ymid) ] )
                cv2.drawContours(image_dot, [triangle_points], 0, (255,255,255), -1)  
                # mirror left right for consistent shape
                image_dot[:, :xmid] = np.fliplr(image_dot[:, xmid+1:])
                # mirror up down to create diamond shape
                image_dot[ymid:, :] = np.flipud(image_dot[:ymid+1, :])
                    
            # rotate dot image
            dot_matrix_rotate_value = random.randint(self.dot_matrix_rotate_value_range[0], self.dot_matrix_rotate_value_range[1])
            if dot_matrix_rotate_value!=0:  
                image_dot_PIL = Image.fromarray(image_dot)  
                rotated_image_dot_PIL= image_dot_PIL.rotate(dot_matrix_rotate_value)
                image_dot = np.array(rotated_image_dot_PIL)
                       
            # apply Gaussian Blur on dot image
            dot_matrix_gaussian_kernel_value = random.randint(self.dot_matrix_gaussian_kernel_value_range[0], self.dot_matrix_gaussian_kernel_value_range[1])
            # Gaussian kernel value must be odd 
            if not dot_matrix_gaussian_kernel_value % 2:
                dot_matrix_gaussian_kernel_value += 1
            image_dot = cv2.GaussianBlur(image_dot,(dot_matrix_gaussian_kernel_value,dot_matrix_gaussian_kernel_value),0)
 
            # divided value from each dot width and height
            div_x = xsize / dot_matrix_dot_width
            div_y = ysize / dot_matrix_dot_height
            
            # number of divisible width and height
            n_dot_x = int(np.floor(div_x))
            n_dot_y = int(np.floor(div_y))
            
            # remainder of division by dot width and height
            remainder_x= xsize % dot_matrix_dot_width
            remainder_y= ysize % dot_matrix_dot_height
            
            image_dot_matrix = np.zeros((ysize, xsize, 3), dtype="uint8")
            # fill in image_dot
            for y in range(n_dot_y):
                cy = y * dot_matrix_dot_height
                for x in range(n_dot_x):
                    cx = x * dot_matrix_dot_width
                    # mean of current dot color
                    dot_color = np.mean(image[cy:cy+dot_matrix_dot_height, cx:cx+dot_matrix_dot_width],axis=(0,1))
                    # multiply to mask of shape
                    image_dot_color = ((image_dot/255) * dot_color).astype("uint8")  
                    # apply dot to image
                    image_dot_matrix[cy:cy+dot_matrix_dot_height, cx:cx+dot_matrix_dot_width] = image_dot_color
                       
            # remaining last column
            if remainder_y > 0:
                for x in range(n_dot_x):
                    cx = x * dot_matrix_dot_width
                    start_y = n_dot_y *dot_matrix_dot_height
                    # mean of current dot color
                    dot_color = np.mean(image[start_y:start_y+remainder_y , cx:cx+dot_matrix_dot_width],axis=(0,1))
                    # multiply to mask of shape
                    image_dot_color = ((image_dot/255) * dot_color).astype("uint8")  
                    # apply dot to image
                    image_dot_matrix[start_y:start_y+remainder_y , cx:cx+dot_matrix_dot_width] = image_dot_color[:remainder_y, :]
      
            # remaining last row
            if remainder_x > 0:
                for y in range(n_dot_y):
                    cy = y * dot_matrix_dot_height
                    start_x = n_dot_x *dot_matrix_dot_width
                    # mean of current dot color
                    dot_color = np.mean(image[cy:cy+dot_matrix_dot_height, start_x:start_x+remainder_x],axis=(0,1))
                    # multiply to mask of shape
                    image_dot_color = ((image_dot/255) * dot_color).astype("uint8")  
                    # apply dot to image
                    image_dot_matrix[cy:cy+dot_matrix_dot_height, start_x:start_x+remainder_x]  = image_dot_color[:, :remainder_x]
     
            # last dot (bottom right)
            if remainder_x and remainder_y > 0:
                
                if remainder_x >0:
                    length_x = remainder_x
                else:
                    length_x = dot_matrix_dot_width
                if remainder_y >0:
                    length_y = remainder_y
                else:
                    length_y = dot_matrix_dot_height
                    
                start_x = n_dot_x *dot_matrix_dot_width
                start_y = n_dot_y *dot_matrix_dot_height
                    
                # mean of current dot color
                dot_color = np.mean(image[start_y:start_y+length_y, start_x:start_x+length_x],axis=(0,1))
                # multiply to mask of shape
                image_dot_color = ((image_dot/255) * dot_color).astype("uint8")  
                image_dot_matrix[start_y:start_y+length_y, start_x:start_x+length_x]  = image_dot_color[:length_y, :length_x]
                    
            # return image follows the input image color channel
            if is_gray:
                image_dot_matrix = cv2.cvtColor(image_dot_matrix, cv2.COLOR_BGR2GRAY)

            return image_dot_matrix

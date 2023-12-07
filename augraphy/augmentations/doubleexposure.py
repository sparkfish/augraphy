import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation



class DoubleExposure(Augmentation):
    """Emulates double exposure effect when taking picture with phone camera.

    :param gaussian_kernel_range: Pair of ints determining the value of Gaussian kernel.
    :type gaussian_kernel_range: tuple, optional
    :param offset_direction: Direction of exposure effect.
        Use "random: for random direction.
        Use 0 for horizontal direction and 1 for vertical direction.
    :type offset_direction: string or int, optional
    :param offset_range: Tuple of ints determining the value of exposure offset.
    :type offset_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        gaussian_kernel_range=(9,12),
        offset_direction=1,
        offset_range=(18,25),
        p=1,
    ):
        super().__init__(p=p)
        self.gaussian_kernel_range = gaussian_kernel_range
        self.offset_direction = offset_direction
        self.offset_range = offset_range
         

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DoubleExposure(gaussian_kernel_range={self.gaussian_kernel_range}, offset_direction={self.offset_direction}, offset_range={self.offset_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # check and convert image into BGR format
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


            # generate random gaussian kernel
            gaussian_kernel = random.randint(self.gaussian_kernel_range[0], self.gaussian_kernel_range[1])
             # kernel must be odd
            if not gaussian_kernel % 2:
                gaussian_kernel += 1

            # generate offset
            offset = random.randint(self.offset_range[0], self.offset_range[1])
            if offset > 0:
                sign = 1
            else:
                sign = -1
            
            # update offset direction
            if self.offset_direction == "random":
                offset_direction = random.choice([0,1])
            else:
                offset_direction = self.offset_direction

            # apply blur
            image_blur = cv2.GaussianBlur(image, [gaussian_kernel,gaussian_kernel],0)
            image_output = image_blur.copy()
            if offset != 0:
                for i in range(1, abs(offset)+1):
                    coffset = i
                    image_blur_offset = image_blur.copy()
                    # vertical
                    if offset_direction:  
                        if sign>0:
                            image_blur_offset[coffset:,:] = image_blur_offset[:-coffset,:] 
                        else:
                            image_blur_offset[:-coffset,:] = image_blur_offset[coffset:,:] 
                   # horizontal
                    else:
                        if sign>0:
                            image_blur_offset[:, coffset:] = image_blur_offset[:, :-coffset] 
                        else:
                            image_blur_offset[:, :-coffset] = image_blur_offset[:, coffset:] 
                       
                    # blend blur image to output
                    image_output = cv2.addWeighted(image_output, 0.9, image_blur_offset, 0.1, 0)
                        
            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_output] + outputs_extra
            else:
                return image_output

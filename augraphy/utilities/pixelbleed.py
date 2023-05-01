"""
version: 0.0.1
*********************************

Dependencies:
- numpy
- opencv

*********************************


References

- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation: https://docs.opencv.org/4.x/

*********************************

"""
import random

import cv2
import numpy as np


class PixelBleed:
    """

    :param bleed_type: randomly assigns the type of pixel color distortion method if is set to "random" or None
                       RGB pixel color distortion is choosen if "RGB" is passed as a parameter value
                       CMYK pixel color distortion is choosen if "CMYK" is passed as a parameter value
                       By default the value of bleed_type is set to "random"
    :type bleed_type: string, optional

    :param nc: number of color channels, by default the number of color channels is set to 3
    :type nc: int, optional

    """

    def __init__(self, bleed_type="random", nc=3):
        # randomization is done in the init func as it is a utility function
        if bleed_type == "random" or bleed_type is None:
            self.bleed_type = random.choice(["RGB", "CMYK"])
        else:
            self.bleed_type = "RGB" if bleed_type == "RGB" else "CMYK"
        self.nc = nc

    def get_new_val(self, old_val):
        return np.round(old_val * (self.nc - 1)) / (self.nc - 1)

    def __call__(self, image):
        # print(self.bleed_amount, self.bleed_type)
        if self.bleed_type == "RGB":
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Otsu's thresholding to create a binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Apply a morphological closing operation to fill in gaps in the text regions
            kernel = np.ones((4, 4), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            # Create a mask by subtracting the closed image from the binary threshold image
            mask = thresh - closed
            # Splitting the image into RGB channels
            r, g, b = cv2.split(image)
            # Apply the mask as a threshold to each channel
            r_masked = cv2.bitwise_and(r, mask)
            g_masked = cv2.bitwise_and(g, mask)
            b_masked = cv2.bitwise_and(b, mask)
            # define a constant to add to the RGB channel
            bleed_amount = random.choice([50, 60, 70, 80, 90, 100, 110, 120, 150])
            # Add the bleed amount to each masked pixel value
            r_bled = np.clip(r_masked + bleed_amount, 0, 255)
            g_bled = np.clip(g_masked + bleed_amount, 0, 255)
            b_bled = np.clip(b_masked + bleed_amount, 0, 255)
            # Merge the channels back into an RGB image
            image_bled = cv2.merge([r_bled, g_bled, b_bled])
            # Combine the masked and bled images to get the final result
            result = cv2.bitwise_or(image_bled, image)
        if self.bleed_type == "CMYK":
            image = np.array(image, dtype=float) / 255
            arr = self.get_new_val(image)
            result = np.array(arr / np.max(arr) * 255, dtype=np.uint8)
        return result

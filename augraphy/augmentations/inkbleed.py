import random
import sys

import cv2
import numpy as np

from augraphy.augmentations.lib import add_noise
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation


class InkBleed(Augmentation):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the intensity of the
           ink bleeding effect.
    :type intensity: tuple, optionall
    :param kernel_size: Kernel size to determine area of inkbleed effect.
    :type kernel_size: tuple, optional
    :param severity: Severity to determine concentration of inkbleed effect.
    :type severity: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        intensity_range=(0.4, 0.7),
        kernel_size=(5, 5),
        severity=(0.3, 0.4),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.kernel_size = kernel_size
        self.severity = severity

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"InkBleed(intensity_range={self.intensity_range}, kernel_size={self.kernel_size}, severity={self.severity}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            # convert and make sure image is color image
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            image_output = image.copy()

            # apply sobel filter and dilate image
            sobelized = sobel(image)
            kernel = np.ones(self.kernel_size, dtype="uint8")
            sobelized_dilated = cv2.dilate(sobelized, kernel, iterations=1)

            # create grayscale fromn the dilated edge image
            sobelized_dilated_gray = cv2.cvtColor(sobelized_dilated, cv2.COLOR_BGR2GRAY)

            # dilation on the darker ink area, which is erosion here
            dilated = cv2.erode(image, kernel, iterations=1)

            # create a random mask
            image_random = np.random.randint(0, 255, size=image.shape[:2]).astype("uint8")

            # based on the provided severity value, update image edges randomly into the dilated edge image
            severity = random.uniform(self.severity[0], self.severity[1]) * 255
            indices = np.logical_and(image_random < severity, sobelized_dilated_gray > 0)
            image_output[indices] = dilated[indices]

            # blur image and blend output based on input intensity
            image_output = cv2.GaussianBlur(image_output, (3, 3), 0)
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            image_output = cv2.addWeighted(image_output, intensity, image, 1 - intensity, 0)

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

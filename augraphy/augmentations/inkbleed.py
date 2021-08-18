import cv2
import numpy as np
import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class InkBleed(Augmentation):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the range from which
           noise intensity is sampled.
    :type intensity: tuple, optional
    :param color_range: Pair of ints determining the range from which color
           noise is sampled.
    :type color_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self, intensity_range=(0.1, 0.2), color_range=(0, 224), p=0.5
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_range = color_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"InkBleed(intensity_range={self.intensity_range}, color_range={self.color_range}, p={self.p})"

    def sobel(self, image):
        """Computes the gradient of the image intensity function.

        :param image: The image over which to create an edge mask.
        :type image: numpy.array
        """
        gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["ink"][-1].result
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            add_noise_fn = (
                lambda x, y: random.randint(self.color_range[0], self.color_range[1])
                if (y == 255 and random.random() < intensity)
                else x
            )
            apply_mask_fn = (
                lambda x, y, z: x if (z != 255 or x < 64) else y
            )
            add_noise = np.vectorize(add_noise_fn)
            apply_mask = np.vectorize(apply_mask_fn)
            sobel = self.sobel(image)
            sobel = cv2.dilate(sobel, (3,3), iterations=1)
            noise_mask = add_noise(image, sobel)
            noise_mask = noise_mask.astype("uint8")
            noise_mask = cv2.GaussianBlur(noise_mask, (3,3), 0)
            
            image = apply_mask(image, noise_mask, sobel)

            data["ink"].append(AugmentationResult(self, image))

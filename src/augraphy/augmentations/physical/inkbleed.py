import cv2
import numpy as np
import random

from base.augmentation import Augmentation
from base.augmentationResult import AugmentationResult



class InkBleedAugmentation(Augmentation):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the range from which
           noise intensity is sampled.
    :type intensity: tuple, optional
    :param color_range: Pair of ints determining the range from which color
           noise is sampled.
    :type color_range: tuple, optional
    :param probability: The probability this Augmentation will be applied.
    :type probability: float, optional
    """
    def __init__(
        self, intensity_range=(0.1, 0.2), color_range=(0, 224), probability=0.5
    ):
        """Constructor method"""
        super().__init__(probability=probability)
        self.intensity_range = intensity_range
        self.color_range = color_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"InkBleedAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, probability={self.probability})"


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
            img = data["ink"][-1].result
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            add_noise_fn = (
                lambda x, y: random.randint(self.color_range[0], self.color_range[1])
                if (y == 255 and random.random() < intensity)
                else x
            )
            add_noise = np.vectorize(add_noise_fn)
            sobel = self.sobel(img)
            img = add_noise(img, sobel)

            data["ink"].append(AugmentationResult(self, img))

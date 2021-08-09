import cv2
import numpy as np
import random

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult

class BrightnessTexturizeAugmentation(Augmentation):
    """Creates a random noise in the brightness channel to emulate paper
    textures.

    :param range: Pair of ints determining the range from which to sample values
           for the brightness matrix.
    :type range: tuple, optional
    :param deviation: Additional variation for the uniform sample.
    :type deviation: float, optional
    :param layer: The image layer to apply the brightness texturization to
    :type layer: string, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self, range=(0.9, 0.99), deviation=0.03, layer="paper", p=0.5
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.low = range[0]
        self.high = range[1]
        self.deviation = deviation
        self.layer = layer
        self.range = range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BrightnessTexturizeAugmentation(layer='{self.layer}', range={self.range}, deviation={self.deviation}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = data[self.layer][-1].result
            value = random.uniform(self.low, self.high)
            hsv = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)

            low = value - (value * self.deviation)  # *random.uniform(0, deviation)
            max = value + (value * self.deviation)

            makerand = np.vectorize(lambda x: random.uniform(low,max))
            brightness_matrix = makerand(np.zeros((hsv.shape[0], hsv.shape[1])))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            hsv = np.array(hsv, dtype=np.uint8)
            hsv = cv2.bitwise_not(hsv)
            hsv = np.array(hsv, dtype=np.float64)

            low = value - (value * self.deviation)
            max = value + (value * self.deviation)

            makerand = np.vectorize(lambda x: random.uniform(low,max))
            brightness_matrix = makerand(np.zeros((hsv.shape[0], hsv.shape[1])))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            hsv = np.array(hsv, dtype=np.uint8)
            hsv = cv2.bitwise_not(hsv)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            data[self.layer].append(AugmentationResult(self, img))

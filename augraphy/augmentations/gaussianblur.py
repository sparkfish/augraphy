import cv2
import random
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class GaussianBlur(Augmentation):
    """Applies a Gaussian blur to the whole image.

    :param layer: The image layer to apply the blur to.
    :type layer: string
    :param kernels: List of height/width pairs from which to sample the kernel
           size.
    :type kernels: list, optional
    :param sigmaX: Standard deviation of the kernel along the x-axis.
    :type sigmaX: float, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, layer, kernels=[(3, 3)], sigmaX=0, p=0.5):
        """Constructor method"""
        super().__init__(p=p)
        self.sigmaX = sigmaX
        self.kernels = kernels
        self.layer = layer

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            img = np.double(data[self.layer][-1].result)
            img = cv2.GaussianBlur(img, random.choice(self.kernels), self.sigmaX)
            data[self.layer].append(AugmentationResult(self, img))

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"GaussianBlur({self.layer}, kernels={self.kernels}, sigmaX={self.sigmaX}, p={self.p})"

import random

import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Scale(Augmentation):
    """Scales the image up or down.

    The OpenCV recommended algorithm is chosen based on the scaling factor
    supplied. Factors less than 1 will shrink the image, factors greater than 1
    will zoom the image.

    :param layer: The pipeline layer this augmentation will be applied in.
    :type layer: string
    :param scale_factor: How much to scale the image by.
    :type scale_factor: float, optional
    :param interpolation: The interpolation algorithm to use when scaling.
    :type interpolation: cv2 interpolation flag
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, layer, scale_factor, interpolation=None, p=0.5):
        """Constructor method"""
        super().__init__(p=p)
        self.layer = layer
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Scale(layer={self.layer}, scale_factor={self.scale_factor}, interpolation={self.interpolation}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result.copy()

            new_size = (
                round(image.shape[1] * self.scale_factor),
                round(image.shape[0] * self.scale_factor),
            )

            if self.scale_factor <= 1:
                if self.interpolation is None:
                    scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                else:
                    scaled = cv2.resize(
                        image,
                        new_size,
                        interpolation=self.interpolation,
                    )
            else:
                if self.interpolation is None:
                    scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
                else:
                    scaled = cv2.resize(
                        image,
                        new_size,
                        interpolation=self.interpolation,
                    )

            data[self.layer].append(AugmentationResult(self, scaled))

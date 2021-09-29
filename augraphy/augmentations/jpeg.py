import random

import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Jpeg(Augmentation):
    """Uses JPEG encoding to create compression artifacts in the image.

    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param quality_range: Pair of ints determining the range from which to
           sample the compression quality.
    :type quality_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        quality_range=(25, 95),
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.layer = layer
        self.quality_range = quality_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Jpeg({self.layer}, quality_range={self.quality_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result.copy()
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                random.randint(self.quality_range[0], self.quality_range[1]),
            ]
            result, encimg = cv2.imencode(".jpg", image, encode_param)
            image = cv2.imdecode(encimg, 1)
            data[self.layer].append(AugmentationResult(self, image))

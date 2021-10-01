import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class NoiseTexturize(Augmentation):
    """Creates a random noise based texture pattern to emulate paper textures.
    Consequently applies noise patterns to the original image from big to small.

    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param sigma_range: Defines bounds of noise fluctuations.
    :type sigma_range: tuple, optional
    :param turbulence_range: Defines how quickly big patterns will be
        replaced with the small ones. The lower value -
        the more iterations will be performed during texture generation.
    :type turbulence_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        layer,
        sigma_range=(3, 10),
        turbulence_range=(2, 5),
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.layer = layer
        self.sigma_range = sigma_range
        self.turbulence_range = turbulence_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"NoiseTexturize(layer={self.layer}, sigma_range={self.sigma_range}, turbulence_range={self.turbulence_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result.copy()

            sigma = random.randint(self.sigma_range[0], self.sigma_range[1])
            turbulence = random.randint(
                self.turbulence_range[0],
                self.turbulence_range[1],
            )

            result = image.astype(float)
            cols, rows, ch = image.shape
            ratio = cols
            while not ratio == 1:
                result += self.noise(cols, rows, ratio, sigma=sigma)
                ratio = (ratio // turbulence) or 1
            cut = np.clip(result, 0, 255)

            data[self.layer].append(AugmentationResult(self, cut.astype(np.uint8)))

    def noise(self, width, height, ratio, sigma):
        """The function generates an image, filled with gaussian nose. If ratio
        parameter is specified, noise will be generated for a lesser image and
        then it will be upscaled to the original size. In that case noise will
        generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param ratio: the size of generated noise "pixels"
        :param sigma: defines bounds of noise fluctuations
        """
        mean = 0
        # assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
        # assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

        h = int(height / ratio)
        w = int(width / ratio)

        if h == 0:
            h = 1
        if w == 0:
            w = 1

        gaussian = np.vectorize(lambda x: random.gauss(mean, sigma))
        result = gaussian(np.array((w, h)))

        if ratio > 1:
            result = cv2.resize(
                result,
                dsize=(width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return result.reshape((width, height, 1))

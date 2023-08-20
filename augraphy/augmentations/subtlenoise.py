import random

import numpy as np

from augraphy.base.augmentation import Augmentation


class SubtleNoise(Augmentation):
    """Emulates the imperfections in scanning solid colors due to subtle
    lighting differences.

    :param subtle_range: The possible range of noise variation to sample from.
    :type subtle_range: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        subtle_range=10,
        p=1,
    ):
        super().__init__(p=p)
        self.subtle_range = subtle_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"SubtleNoise(subtle_range={self.subtle_range}, p={self.p})"

    def add_subtle_noise(self, image):
        """Generate mask of noise and add it to input image.

        :param image: Image to apply the function.
        :type image: numpy.array (numpy.uint8)
        """
        # get image size
        ysize, xsize = image.shape[:2]

        # generate 2d mask of random noise
        image_noise = np.random.randint(-self.subtle_range, self.subtle_range, size=(ysize, xsize))

        # add noise to image
        image = image.astype("int") + image_noise

        return image

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # multiple channels image
            if len(image.shape) > 2:
                # convert to int to enable negative
                image = image.astype("int")
                # skip alpha layer
                for i in range(3):
                    image[:, :, i] = self.add_subtle_noise(image[:, :, i])
            # single channel image
            else:
                image = self.add_subtle_noise(image)

            # clip values between 0-255
            image = np.clip(image, 0, 255)

            return image.astype("uint8")

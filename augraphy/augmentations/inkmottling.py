import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class InkMottling(Augmentation):
    """Create a random pattern effect in the detected ink by blending a layer of random Gaussian noise.

    :param ink_mottling_alpha_range: Tuple of floats determining the alpha value of the added effect.
    :type ink_mottling_alpha_range: tuple, optional
    :param ink_mottling_noise_scale_range: Tuple of ints determining the size of Gaussian noise pattern.
    :type ink_mottling_noise_scale_range: tuple, optional
    :param ink_mottling_gaussian_kernel_range: Tuple of ints determining the Gaussian kernel value.
    :type ink_mottling_gaussian_kernel_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        ink_mottling_alpha_range=(0.2, 0.3),
        ink_mottling_noise_scale_range=(2, 2),
        ink_mottling_gaussian_kernel_range=(3, 5),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.ink_mottling_alpha_range = ink_mottling_alpha_range
        self.ink_mottling_noise_scale_range = ink_mottling_noise_scale_range
        self.ink_mottling_gaussian_kernel_range = ink_mottling_gaussian_kernel_range

    def __repr__(self):
        return f"InkMottling(ink_mottling_alpha_range={self.ink_mottling_alpha_range}, ink_mottling_noise_scale_range={self.ink_mottling_noise_scale_range}, ink_mottling_gaussian_kernel_range={self.ink_mottling_gaussian_kernel_range}, p={self.p})"

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            ysize, xsize = image.shape[:2]

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            image_mask = np.full((ysize, xsize), fill_value=0, dtype="uint8")

            # get ink area from each channel
            for i in range(3):
                #  convert image into binary
                _, image_binary = cv2.threshold(image[:, :, i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                image_mask += image_binary

            # invert ink (dark area)
            image_mask = 255 - image_mask

            # generate random mask of noise
            ink_mottling_noise_scale = random.randint(
                self.ink_mottling_noise_scale_range[0],
                self.ink_mottling_noise_scale_range[1],
            )
            image_random = np.random.randint(
                0,
                255,
                size=(int(ysize / ink_mottling_noise_scale), int(xsize / ink_mottling_noise_scale)),
            ).astype("uint8")
            image_random = cv2.cvtColor(image_random, cv2.COLOR_GRAY2BGR)

            # apply gaussian blur to the mask of noise
            kernel_value = random.randint(
                self.ink_mottling_gaussian_kernel_range[0],
                self.ink_mottling_gaussian_kernel_range[1],
            )
            # kernel must be odd
            if not (kernel_value % 2):
                kernel_value += 1
            image_random = cv2.GaussianBlur(image_random, (kernel_value, kernel_value), 0)

            # resize to input image size
            if ink_mottling_noise_scale > 1:
                image_random = cv2.resize(
                    image_random,
                    (xsize, ysize),
                    interpolation=cv2.INTER_AREA,
                )

            # blend noise mask with image ink based on the input alpha
            ink_mottling_alpha = random.uniform(self.ink_mottling_alpha_range[0], self.ink_mottling_alpha_range[1])
            image_blend = cv2.addWeighted(image, (1 - ink_mottling_alpha), image_random, ink_mottling_alpha, 0)
            image[image_mask > 128] = image_blend[image_mask > 128]

            from matplotlib import pyplot as plt

            plt.figure()
            plt.imshow(image_mask)

            plt.figure()
            plt.imshow(image_blend)

            # return image follows the input image color channel
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image

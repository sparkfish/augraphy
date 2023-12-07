import random

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class Moire(Augmentation):
    """Creates a moire pattern effect in the image by blending the moire pattern using OverlayBuilder.

    :param moire_density: Pair of ints determining of density of the moire pattern stripes.
    :type moire_density: tuple, optional
    :param moire_blend_method: The blending method to blend moire pattern into the input image.
    :type moire_blend_method: int, optional
    :param moire_blend_alpha: The blending alpha value for blending method with the usage of alpha.
    :type moire_blend_alpha: float, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        moire_density=(15, 20),
        moire_blend_method="normal",
        moire_blend_alpha=0.1,
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.moire_density = moire_density
        self.moire_blend_method = moire_blend_method
        self.moire_blend_alpha = moire_blend_alpha
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Moire(moire_density={self.moire_density}, moire_blend_method={self.moire_blend_method}, moire_blend_alpha={self.moire_blend_alpha}, numba_jit={self.numba_jit}, p={self.p})"

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def generate_moire_pattern(xsize, ysize, density_range):
        """Generate moire pattern by using sine function.

        :param xsize: Width of generated moire pattern.
        :type xsize: int, optional
        :param ysize: Height of generated moire pattern.
        :type ysize: int, optional
        :param density_range: Pair of ints determining of density of the moire pattern stripes.
        :type density_range: tuple, optional
        """

        image = np.zeros((ysize, xsize), dtype="uint8")

        # random relative location
        relative_x = random.randint(5, 10)
        relative_y = random.randint(5, 10)

        # random density
        density = random.randint(density_range[0], density_range[1])

        # random phase
        phase = 2 * np.pi * random.uniform(0.001, 0.01)

        # random offset
        if random.random() > 0.5:
            x_offset = random.randint(-5, -2)
        else:
            x_offset = random.randint(2, 5)
        if random.random() > 0.5:
            y_offset = random.randint(-5, -2)
        else:
            y_offset = random.randint(2, 5)

        # create moire pattern
        for y in nb.prange(ysize):
            new_y = ((y / ysize) * (y_offset * relative_y)) - relative_y
            for x in nb.prange(xsize):
                new_x = ((x / xsize) * (x_offset * relative_x)) - relative_x

                value = np.sin(phase + (density * 2 * np.pi * (np.sqrt(new_x**2 + new_y**2))))
                image[x, y] = int(255 * (value + 1) / 2)

        return image

    def blend_moire(self, image, image_moire):
        """Blend moire pattern into the image by using OverLayBuilder.

        :param image: The input image.
        :type image: numpy array, optional
        :param image_moire: Image with generated moire pattern.
        :type image_moire: numpy array, optional
        """

        # minimum intensity so that pattern will not be too dark
        image_moire[image_moire < 30] = 30

        # Create overlay object and blend moire pattern
        ob = OverlayBuilder(
            self.moire_blend_method,
            image_moire,
            image,
            1,
            (1, 1),
            "center",
            self.moire_blend_alpha,
        )

        image_output = ob.build_overlay()

        return image_output

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

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

            # create moire pattern
            image_moire1 = self.generate_moire_pattern(1000, 1000, self.moire_density)
            image_moire2 = self.generate_moire_pattern(1000, 1000, self.moire_density)
            # Create overlay object and blend moire pattern
            ob = OverlayBuilder(
                "overlay",
                image_moire1,
                image_moire2,
                1,
                (1, 1),
                "center",
                0.5,
            )
            image_moire = ob.build_overlay()
            image_moire = cv2.resize(image_moire, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            # enhance effect by using median filter
            image_moire = cv2.medianBlur(image_moire, 5)

            # blend moire pattern into image
            image_output = self.blend_moire(image, image_moire)

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

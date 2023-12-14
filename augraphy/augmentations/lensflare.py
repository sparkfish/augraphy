import math
import random

import cv2
import numba
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import color_modifier
from augraphy.augmentations.lib import lensflare
from augraphy.augmentations.lib import lnorm
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class LensFlare(Augmentation):
    """Creates a lens flare effect by drawing a bright spot light with darker background.

    :param lens_flare_location: Location (x,y) for the lens flare spot light location.
        Use "random" for random location.
    :type lens_flare_location: tuple, optional
    :param lens_flare_color: Color of lens flare effect in BGR.
    :type lens_flare_color: tuple, optional
    :param lens_flare_size: Tuple of floats in determining the size of the lens flare spot light.
    :type lens_flare_size: tuple, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        lens_flare_location="random",
        lens_flare_color="random",
        lens_flare_size=(0.5, 5),
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p, numba_jit=numba_jit)
        self.lens_flare_location = lens_flare_location
        self.lens_flare_color = lens_flare_color
        self.lens_flare_size = lens_flare_size
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LensFlare(lens_flare_location={self.lens_flare_location}, lens_flare_color={self.lens_flare_color}, lens_flare_size={self.lens_flare_size}, numba_jit={self.numba_jit}, p={self.p})"

    # the algorithm is adapted from this link: https://www.shadertoy.com/view/4sX3Rs
    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def generate_lens_flare(image, noise_mask, color_input, light_spot_size, resolution, flare_location):
        """Core function to generate lens flare effect.

        :param image: The canvas of the lens flare effect.
        :type image: numpy array
        :param noise_mask: The mask of noise for randomization.
        :type noise_mask: numpy array
        :param color_input: Color in BGR format.
        :type color_input: numpy array
        :param light_spot_size: The size of spot light in lens flare effect.
        :type light_spot_size: float
        :param resolution: The resolution of the lens flare effect.
        :type resoltuion: numpy array
        :param flare_location: The location of the lens flare effect.
        :type flare_location: numpy array

        """

        for y in numba.prange(image.shape[0]):
            for x in numba.prange(image.shape[1]):
                # scale with resolution
                uv = np.array([x / resolution[0] - 0.5, y / resolution[1] - 0.5], dtype="float")
                # fix aspect ratio
                uv[0] *= resolution[0] / resolution[1]

                # get different in location
                main = uv - flare_location
                uvd = uv * (lnorm(uv))

                # generate distant
                ang = np.arctan2(main[0], main[1])
                dist = lnorm(main)
                dist = np.power(dist, 0.1)

                # generate initial constant
                f0 = 1.0 / (lnorm(uv - flare_location) * 16.0 + 1.0)

                # generate color
                t = np.sin(ang * 2.0 + flare_location[0]) * 4.0 - np.cos(ang * 3.0 + flare_location[1])
                noise_component = noise_mask[math.trunc(t) % 1, 0]
                f0 = f0 + f0 * (np.sin(noise_component * 16.0) * 0.1 + dist * 0.1 + 0.8)

                color = color_input * lensflare(uv, flare_location, uvd, f0)

                # further process color
                noise_component2 = noise_mask[math.trunc(y) % 1, math.trunc(x) % 1]
                color -= noise_component2 * 0.015
                color = color_modifier(color, light_spot_size, 0.1)

                # update color
                image[y, x] = color

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()
            ysize, xsize = image.shape[:2]

            # check and convert image into BGR format
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # get lens flare location
            if self.lens_flare_location == "random":
                flare_location = np.array([random.randint(0, xsize - 1), random.randint(0, ysize - 1)]).astype("float")
            else:
                flare_location = np.array(self.lens_flare_location).astype("float")

            # get color
            if self.lens_flare_color == "random":
                color = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], dtype="float")
            else:
                color = np.array(
                    [self.lens_flare_color[0] / 255, self.lens_flare_color[1] / 255, self.lens_flare_color[2] / 255],
                    dtype="float",
                )

            # get lens flare size
            light_spot_size = random.uniform(self.lens_flare_size[0], self.lens_flare_size[1])

            # fixed internal size
            cysize, cxsize = 100, 100

            # update flare location based on internal size ratio
            xratio = cxsize / xsize
            yratio = cysize / ysize
            flare_location[0] = flare_location[0] * xratio
            flare_location[1] = flare_location[1] * yratio

            # generate empty image
            image_lens_flare = np.full((cysize, cxsize, 3), fill_value=0.3, dtype="float")

            # Get image resolution
            resolution = np.array([cxsize, cysize])

            # mask of random noise
            noise_mask_input = np.random.rand(cysize, cxsize).astype("float")

            # location of flare
            flare_location = np.array(
                [flare_location[0] / resolution[0] - 0.5, flare_location[1] / resolution[1] - 0.5],
            ).astype("float")

            # generate lens flare effect
            self.generate_lens_flare(
                image_lens_flare,
                noise_mask_input,
                color,
                light_spot_size,
                resolution,
                flare_location,
            )

            # convert to uint8
            image_lens_flare[image_lens_flare > 1] = 1
            image_lens_flare = (image_lens_flare * 255).astype(np.uint8)

            # resize into original size
            image_lens_flare = cv2.resize(image_lens_flare, (xsize, ysize), cv2.INTER_LINEAR)

            # blend points image into input again
            ob = OverlayBuilder(
                "ink_to_paper",
                image_lens_flare,
                image,
                1,
                (1, 1),
                "center",
                0,
                0.5,
            )

            # blend lens flare with input image
            image_output = ob.build_overlay()

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

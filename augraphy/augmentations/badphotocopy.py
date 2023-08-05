import random

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import smooth
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.noisegenerator import NoiseGenerator


class BadPhotoCopy(Augmentation):
    """Uses added noise to generate an effect of dirty copier.

    :param mask: Mask of noise to generate badphotocopy effect.
    :type mask: uint8, optional
    :param noise_type: Types of noises to generate different mask patterns. Use -1 to select randomly.
    :type noise_type: int, optional
    :param noise_side: Location of noise.
    :type noise_side: string, optional
    :param noise_iteration: Pair of ints to determine number of iterations to apply noise in the mask.
    :type noise_iteration: tuple, optional
    :param noise_size: Pair of ints to determine scale of noise in the mask.
    :type noise_size: tuple, optional
    :param noise_value: Intensity range of noise, lower value get darker effect.
    :type noise_value: tuple, optional
    :param noise_sparsity: Pair of floats determining sparseness of noise.
    :type noise_sparsity: tuple, optional
    :param noise_concentration: Pair of floats determining concentration of noise.
    :type noise_concentration: tuple, optional
    :param blur_noise: Flag to enable blur in noise mask. Use -1 to select randomly.
    :type blur_noise: int, optional
    :param blur_noise_kernel: Kernel to blur noise mask.
    :type blur_noise_kernel: tuple, optional
    :param wave_pattern: To enable wave pattern in noise. Use -1 to select randomly.
    :type wave_pattern: int, optional
    :param edge_effect: To add sobel edge effect into the noise mask. Use -1 to select randomly.
    :type edge_effect: int, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        mask=None,
        noise_type=-1,
        noise_side="random",
        noise_iteration=(2, 5),
        noise_size=(1, 3),
        noise_value=(0, 64),
        noise_sparsity=(0.1, 0.9),
        noise_concentration=(0.1, 0.6),
        blur_noise=-1,
        blur_noise_kernel=(5, 5),
        wave_pattern=-1,
        edge_effect=-1,
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p, numba_jit=numba_jit)
        self.mask = mask
        self.noise_type = noise_type
        self.noise_side = noise_side
        self.noise_iteration = noise_iteration
        self.noise_size = noise_size
        self.noise_value = noise_value
        self.noise_sparsity = noise_sparsity
        self.noise_concentration = noise_concentration
        self.blur_noise = blur_noise
        self.blur_noise_kernel = blur_noise_kernel
        self.wave_pattern = wave_pattern
        self.edge_effect = edge_effect
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

        # clamp values
        # noise value range from 0-255
        self.noise_value = list(self.noise_value)
        self.noise_value[0] = np.clip(self.noise_value[0], 0, 255)
        self.noise_value[1] = np.clip(self.noise_value[1], 0, 255)

        # sparsity range from 0-1
        self.noise_sparsity = list(self.noise_sparsity)
        self.noise_sparsity[0] = np.clip(self.noise_sparsity[0], 0, 1)
        self.noise_sparsity[1] = np.clip(self.noise_sparsity[1], 0, 1)

        # concentration range from 0-1
        self.noise_concentration = list(self.noise_concentration)
        self.noise_concentration[0] = np.clip(self.noise_concentration[0], 0, 1)
        self.noise_concentration[1] = np.clip(self.noise_concentration[1], 0, 1)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BadPhotoCopy(mask={self.mask}, noise_type={self.noise_type}, noise_side={self.noise_side}, noise_iteration={self.noise_iteration}, noise_size={self.noise_size}, noise_value={self.noise_value}, noise_sparsity={self.noise_sparsity}, noise_concentration={self.noise_concentration}, blur_noise={self.blur_noise}, blur_noise_kernel={self.blur_noise_kernel}, wave_pattern={self.wave_pattern}, edge_effect={self.edge_effect}, numba_jit={self.numba_jit}, p={self.p})"

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def fill_wave_mask(img_wave, smooth_points):
        """Fill wavy pattern mask with value.

        :param img_wave: The image of wavy pattern mask.
        :type img_wave: numpy.array
        :param smooth_points: Coordinates of wavy pattern.
        :type smooth_points: numpy.array
        """

        # fill mask by draw wavy line across image
        n = len(smooth_points)
        for i in nb.prange(n):
            x_point, y_point = smooth_points[i]
            img_wave[:y_point, x_point] = 255

    def apply_wave(self, mask, noise_side):
        """applies wavy pattern mask to input mask.

        :param mask: The image to apply the function.
        :type mask: numpy.array (numpy.uint8)
        :param noise_side: The side of the image to apply the wave effect.
        :type noise_side: string
        """

        # rescale mask from 0 to 255
        # prevent zero division
        if np.max(mask) - np.min(mask) == 0:
            divisor = 1
        else:
            divisor = np.max(mask) - np.min(mask)
        mask_rescaled = (((mask - np.min(mask)) / divisor) * 255).astype("uint8")
        mask_ysize, mask_xsize = mask_rescaled.shape
        img_wave = np.zeros_like(mask_rescaled)

        # get mask size measurements
        mask_y_third_quarter_size = int(mask_ysize * 3 / 4)
        mask_y_one_twelve_size = int(mask_ysize / 12)

        # generate points and at least 6 points
        number_points = random.randint(6, 12)
        points_x = np.linspace(0, mask_xsize - 1, number_points)

        points = np.zeros((number_points, 2), dtype="float")
        points[0] = [0, random.uniform(mask_y_one_twelve_size, mask_y_third_quarter_size)]
        points[-1] = [mask_xsize - 1, random.uniform(mask_y_one_twelve_size, mask_y_third_quarter_size)]
        for i in range(1, number_points - 1):
            # points between 1st and last point
            points[i] = [points_x[i - 1], random.uniform(mask_y_one_twelve_size, mask_y_third_quarter_size)]

        # smooth points
        smooth_points = smooth(points, 12).astype("int")
        smooth_points = np.unique(smooth_points, axis=0)

        # fill wave mask
        self.fill_wave_mask(img_wave, smooth_points)

        # smoothen edge of the wave mask
        blur_intensity = random.randint(2, 3)
        gaussian_kernel_value = random.randint(151, 301)
        # kernel must be odd
        if not gaussian_kernel_value % 2:
            gaussian_kernel_value += 1
        ysize_small = max(100, int(mask_ysize / blur_intensity))
        xsize_small = max(100, int(mask_xsize / blur_intensity))
        img_wave = cv2.resize(img_wave, (xsize_small, ysize_small), interpolation=cv2.INTER_LINEAR)
        img_wave = cv2.GaussianBlur(img_wave, (gaussian_kernel_value, gaussian_kernel_value), 0)
        img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_LINEAR)

        # top (noise concentrated at top edge)
        if noise_side == "top":
            mask = 255 - cv2.multiply(img_wave, 255 - mask, scale=1 / 255)
        # right (noise concentrated at right edge)
        elif noise_side == "right":
            img_wave = np.rot90(img_wave, 3)
            img_wave = cv2.resize(
                img_wave,
                (mask_xsize, mask_ysize),
                interpolation=cv2.INTER_AREA,
            )
            mask = 255 - cv2.multiply(img_wave, 255 - mask, scale=1 / 255)
        # bottom (noise concentrated at bottom edge)
        elif noise_side == "bottom":
            img_wave = np.flipud(img_wave)
            mask = 255 - cv2.multiply(img_wave, 255 - mask, scale=1 / 255)
        # left (noise concentrated at left edge)
        elif noise_side == "left":
            img_wave = np.rot90(img_wave, 1)
            img_wave = cv2.resize(
                img_wave,
                (mask_xsize, mask_ysize),
                interpolation=cv2.INTER_AREA,
            )
            mask = 255 - cv2.multiply(img_wave, 255 - mask, scale=1 / 255)

        return mask.astype("uint8")

    def apply_augmentation(self, image):
        """applies augmentation to the input image.

        :param image: The image to apply the augmentation.
        :type image: numpy.array (numpy.uint8)
        """

        # convert and make sure image is color image
        if len(image.shape) > 2:
            is_gray = 0
        else:
            is_gray = 1
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # get image dimensions
        ysize, xsize = image.shape[:2]

        if self.noise_side == "random":
            noise_side = random.choice(["left", "top", "right", "bottom"])
        else:
            noise_side = self.noise_side

        # check if provided mask is numpy array
        if isinstance(self.mask, np.ndarray):
            mask = self.mask
            # noise mask needs to be in grayscale form
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # generate mask of noise
        else:
            noise_generator = NoiseGenerator(
                noise_type=self.noise_type,
                noise_side=noise_side,
            )
            mask = noise_generator.generate_noise(
                noise_value=self.noise_value,
                noise_iteration=self.noise_iteration,
                noise_size=self.noise_size,
                noise_sparsity=self.noise_sparsity,
                noise_concentration=self.noise_concentration,
                xsize=xsize,
                ysize=ysize,
            )

        # rescale to 0 -255
        mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255
        if self.noise_value[0] > self.noise_value[1]:
            self.noise_value[0] = self.noise_value[1]

        # resize back to original size
        mask = cv2.resize(mask, (xsize, ysize)).astype("uint8")

        # apply blur to mask of noise
        if self.blur_noise == -1:
            blur_noise = random.choice((0, 1))
        else:
            blur_noise = self.blur_noise
        if blur_noise:
            mask = cv2.GaussianBlur(mask, self.blur_noise_kernel, 0)

        # apply wave pattern to mask
        if self.wave_pattern == -1:
            wave_pattern = random.choice((0, 1))
        else:
            wave_pattern = self.wave_pattern
        if wave_pattern:
            mask = self.apply_wave(mask, noise_side)

        # add dotted noise effect to mask (unsmoothen)
        if not blur_noise:
            noise_mask = np.random.random((ysize, xsize)) * 225
            mask[mask > noise_mask] = 255
        noise_img = mask

        # add blur
        gaussian_kernel = (random.choice([3, 5, 7]), random.choice([3, 5, 7]))
        blurred = cv2.GaussianBlur(noise_img, gaussian_kernel, 0)
        noise_img = cv2.multiply(noise_img, blurred, scale=1 / 255)

        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.multiply(noise_img, result[:, :, i], scale=1 / 255)

        # merge sobel mask and noise mask to image
        if self.edge_effect == -1:
            edge_effect = random.choice((0, 1))
        else:
            edge_effect = self.edge_effect
        if edge_effect:

            # get gray image for sobel purpose
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get edge mask
            image_sobel = sobel(image_gray)
            image_sobel = cv2.GaussianBlur(image_sobel, (3, 3), 0)
            image_sobel[:, :][image_sobel[:, :] % 255 != 0] = 255
            image_sobel = cv2.dilate(image_sobel, (15, 15), iterations=5)
            image_sobel_sobel = sobel(image_sobel)
            image_sobel_sobel = cv2.dilate(image_sobel_sobel, (5, 5), iterations=2)

            # add random noise range from 0-128
            image_random = np.random.randint(0, 128, (image.shape[0], image.shape[1]), dtype="uint8")
            image_random2 = np.random.random((image.shape[0], image.shape[1]))
            indices = np.logical_and(image_sobel_sobel == 255, image_random2 < 0.70)
            image_sobel[indices] = image_random[indices]
            image_sobel = cv2.GaussianBlur(image_sobel, (5, 5), 0)

            image_original = image.copy()
            image_copy = image.copy()

            # apply edge
            result_new = result.copy().astype("int")
            for i in range(3):
                result_new[:, :, i] = result[:, :, i] + image_sobel.astype("int")
            image_original[image_original > result_new] = 0
            result_new[result_new > 255] = 0
            result_new[result_new >= image_copy] = 0
            result = image_original + result_new
            # convert back to uint8
            result = result.astype("uint8")

        # return image follows the input image color channel
        if is_gray:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return result

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = self.apply_augmentation(image)
            return result

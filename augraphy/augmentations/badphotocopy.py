import random

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.noisegenerator import NoiseGenerator


class BadPhotoCopy(Augmentation):
    """Uses added noise to generate an effect of dirty copier.

    :param noise_type: Types of noises to generate different mask patterns.
    :type noise_type: int, optional
    :param noise_value: Intensity range of noise, lower value get darker effect.
    :type noise_value: tuple, optional
    :param noise_sparsity: Pair of floats determining sparseness of noise.
    :type noise_sparsity: tuple, optional
    :param noise_concentration: Pair of floats determining concentration of noise.
    :type noise_concentration: tuple, optional
    :param wave_pattern: To enable wave pattern in noise.
    :type wave_pattern: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        noise_type=0,
        noise_value=(30, 60),
        noise_sparsity=(0.4, 0.6),
        noise_concentration=(0.4, 0.6),
        wave_pattern=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.noise_type = noise_type
        self.noise_value = noise_value
        self.noise_sparsity = noise_sparsity
        self.noise_concentration = noise_concentration
        self.wave_pattern = wave_pattern

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
        return f"BadPhotoCopy(noise_type={self.noise_type}, noise_value={self.noise_value}, noise_sparsity={self.noise_sparsity}, noise_concentration={self.noise_concentration}, wave_pattern={self.wave_pattern}, p={self.p})"

    def apply_wave(self, mask):
        """
        applies wavy pattern mask to input mask
        """

        # rescale mask from 0 to 255
        mask_rescaled = mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255).astype("uint8")
        mask_ysize, mask_xsize = mask_rescaled.shape
        img_wave = np.zeros_like(mask_rescaled)

        # get mask size measurements
        mask_y_third_quarter_size = int(mask_ysize * 3 / 4)
        mask_y_one_twelve_size = int(mask_ysize / 12)
        mask_x_one_twelve_size = int(mask_xsize / 12)

        # generate points and at least 12 points
        number_points = random.randint(6, 12)

        # generate random x location
        x_points = [
            random.randint(mask_x_one_twelve_size, mask_xsize - mask_x_one_twelve_size)
            for _ in range(number_points - 2)
        ]
        x_points.sort()

        # 1st point
        points = [(0, random.randint(mask_y_one_twelve_size, mask_y_third_quarter_size))]
        for i in range(1, number_points - 1, 1):
            # points between 1st and last point
            points.append(
                (x_points[i - 1], random.randint(mask_y_one_twelve_size, mask_y_third_quarter_size)),
            )  # last point
        # last point
        points.append(
            (mask_xsize - 1, random.randint(mask_y_one_twelve_size, mask_y_third_quarter_size)),
        )  # last point

        # smooth points
        smooth_points = smooth(points, 12)

        # draw wavy line across image
        for (x_point, y_point) in smooth_points:
            img_wave[: int(y_point), int(x_point)] = 1
            # additional noise, to smooth the edges between wavy mask
            for y in range(int(y_point), int(mask_ysize - mask_y_one_twelve_size), 1):
                if random.random() > 0.5:
                    img_wave[y, int(x_point)] = 1
                    if random.random() > 0.95:
                        break

        # top (noise concentrated at top edge)
        if self.noise_type == 7:
            mask = img_wave * mask
        # right (noise concentrated at right edge)
        elif self.noise_type == 6:
            img_wave = np.rot90(img_wave, 3)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # bottom (noise concentrated at bottom edge)
        elif self.noise_type == 8:
            img_wave = np.flipud(img_wave)
            mask = img_wave * mask
        # left (noise concentrated at left edge)
        elif self.noise_type == 5:
            img_wave = np.rot90(img_wave, 1)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        else:
            img_wave = np.rot90(img_wave, random.randint(0, 3))
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask

        # reset 0 area to white (white is background)
        mask[img_wave == 0] = 255

        # blur  the mask
        mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)

        return mask

    def apply_augmentation(self, image):

        # get image dimensions
        if len(image.shape) > 2:
            ysize, xsize, dim = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            ysize, xsize = image.shape

        # add random noise range from 0-128
        add_edge_noise_fn = lambda x, y: random.randint(0, 128) if (y == 255 and random.random() < 0.70) else x
        add_edge_noise = np.vectorize(add_edge_noise_fn)

        image_sobel = sobel(image)
        image_sobel = cv2.GaussianBlur(image_sobel, (3, 3), 0)
        image_sobel[:, :][image_sobel[:, :] % 255 != 0] = 255
        image_sobel = cv2.dilate(image_sobel, (5, 5), iterations=2)

        image_sobel_sobel = sobel(image_sobel)
        image_sobel_sobel = cv2.dilate(image_sobel_sobel, (3, 3), iterations=2)
        image_sobel = add_edge_noise(image_sobel, image_sobel_sobel)
        image_sobel = cv2.GaussianBlur(image_sobel, (5, 5), 0)

        # get mask of noise and resize it to image size
        noise_generator = NoiseGenerator(noise_type=self.noise_type)
        mask = noise_generator.generate_noise(
            noise_value=self.noise_value,
            noise_sparsity=self.noise_sparsity,
            noise_concentration=self.noise_concentration,
        )

        # randomly rotate mask
        if self.noise_type != 5 and self.noise_type != 6 and self.noise_type != 7 and self.noise_type != 8:
            mask = np.rot90(mask, random.randint(0, 3))

        # new size after rotation
        mask_ysize, mask_xsize = mask.shape[:2]

        # rescale to 0 -255
        mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255
        if self.noise_value[0] > self.noise_value[1]:
            self.noise_value[0] = self.noise_value[1]

        # creates random small dot of noises
        mask += random.randint(self.noise_value[0], self.noise_value[1])
        mask[mask > 255] = 255
        mask = cv2.resize(mask, (image.shape[1], image.shape[0])).astype("uint8")

        # apply wave pattern to mask
        if self.wave_pattern:
            mask = self.apply_wave(mask)

        # random flip mask vertically or horizontally
        if random.choice([True, False]):
            mask = cv2.flip(mask, 0)
        if random.choice([True, False]):
            mask = cv2.flip(mask, 1)

        # add dotted noise effect to mask (unsmoothen)
        noise_mask = np.random.random((ysize, xsize)) * 225
        mask[mask > noise_mask] = 255
        noise_img = mask

        # add blur
        gaussian_kernel = (random.choice([3, 5, 7]), random.choice([3, 5, 7]))
        blurred = cv2.GaussianBlur(noise_img, gaussian_kernel, 0)
        noise_img = cv2.multiply(noise_img, blurred, scale=1 / 255)
        result = cv2.multiply(noise_img, image, scale=1 / 255)

        # merge sobel mask and noise mask to image
        image_original = image.copy()
        image_copy = image.copy()
        result_new = result.astype("int") + image_sobel.astype("int")
        image_original[image_original > result_new] = 0
        result_new[result_new > 255] = 0
        result_new[result_new > image_copy] = 0
        result = image_original + result_new

        return result.astype("uint8")

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            result = self.apply_augmentation(image)
            return result

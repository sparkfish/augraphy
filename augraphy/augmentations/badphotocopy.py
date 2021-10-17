import random

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation


class BadPhotoCopy(Augmentation):
    """Uses Perlin noise to generate an effect of dirty copier.

    :param noise_type: Types of noises to generate different mask patterns.
    :type noise_type: int, optional
    :param intensity: Intensity range of noise, lower value get darker effect.
    :type intensity: tuple, optional
    :param nperiod: The number of periods of noise to generate along each
                axis.
    :type nperiod: tuple, optional
    :param octaves: The number of octaves in the noise.
    :type octaves: int, optional
    :param persistence:The scaling factor between two octaves.
    :type persistence: int, optional
    :param lacunarity: The frequency factor between two octaves.
    :type lacunarity: int, optional
    :param wave_pattern: To enable wave pattern in noise.
    :type wave_pattern: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        noise_type=0,
        intensity=(30, 60),
        nperiod=(random.randint(1, 4), random.randint(1, 4)),
        octaves=1,
        persistence=0.5,
        lacunarity=2,
        wave_pattern=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.noise_type = noise_type
        self.intensity = intensity
        self.nperiod = nperiod
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.wave_pattern = wave_pattern

        # clamp betweem 0 - 255
        self.intensity = list(self.intensity)
        self.intensity[0] = max(0, min(self.intensity[0], 255))
        self.intensity[1] = max(0, min(self.intensity[1], 255))

        if self.noise_type == 1:
            self.nperiod = (1, 1)
        elif self.noise_type == 2:
            self.nperiod = nperiod = (random.randint(2, 100), random.randint(2, 100))
        elif noise_type == 3:
            self.nperiod = (random.randint(250, 350), random.randint(250, 350))
            self.intensity = (100, 150)
        elif noise_type == 4:
            self.nperiod = (301, 301)
            self.intensity = (200, 250)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BadPhotoCopy(noise_type={self.noise_type}, intensity={self.intensity}, nperiod={self.nperiod}, octaves={self.octaves}, persistence={self.persistence}, lacunarity={self.lacunarity}, wave_pattern={self.wave_pattern}, p={self.p})"

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

        # get noise intensity of each side
        top = np.sum(mask_rescaled[:10, :mask_xsize]) / (10 * mask_xsize * 255)
        right = np.sum(mask_rescaled[:mask_ysize, mask_xsize - 10 : mask_xsize]) / (10 * mask_ysize * 255)
        bottom = np.sum(mask_rescaled[mask_ysize - 10 : mask_ysize, :mask_xsize]) / (10 * mask_xsize * 255)
        left = np.sum(mask_rescaled[:mask_ysize, :10]) / (10 * mask_ysize * 255)

        # get side with highest noise intensity (lower value due to darkest area)
        min_noise = min([top, right, bottom, left])

        # top (noise concentrated at top edge)
        if top == min_noise:
            mask = img_wave * mask
        # top (noise concentrated at right edge)
        elif right == min_noise:
            img_wave = np.rot90(img_wave, 3)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # bottom (noise concentrated at bottom edge)
        elif bottom == min_noise:
            img_wave = np.flipud(img_wave)
            mask = img_wave * mask
        # left (noise concentrated at left edge)
        elif left == min_noise:
            img_wave = np.rot90(img_wave, 1)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask

        # reset 0 area to white (white is background)
        mask[img_wave == 0] = 255

        # blue  the mask
        mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)

        return mask

    def interpolant(self, t):
        """
        Function to perform interpolation
        """
        return 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)

    def perlin_noise_2d(
        self,
        input_size,
        nperiod,
        interpolant,
    ):
        """
        Generate a 2D numpy array of perlin noise.
        """

        # prevent size mismatch issue
        new_size = list(input_size).copy()
        new_size[0] -= input_size[0] % nperiod[0]
        new_size[1] -= input_size[1] % nperiod[1]

        # prevent size <0
        if new_size[0] <= 0 or new_size[1] <= 0:
            return np.zeros((input_size))

        delta = (nperiod[0] / new_size[0], nperiod[1] / new_size[1])
        d = (new_size[0] // nperiod[0], new_size[1] // nperiod[1])
        grid = np.mgrid[0 : nperiod[0] : delta[0], 0 : nperiod[1] : delta[1]].transpose(1, 2, 0) % 1

        # Gradients
        if self.noise_type == 1:
            rperiod = np.zeros((nperiod[0] + 1, nperiod[1] + 1))
            rperiod[1, :] = np.random.rand(1, nperiod[1] + 1)
        elif self.noise_type == 2:
            rperiod = np.zeros((nperiod[0] + 1, nperiod[1] + 1))
            rperiod[1, :] = 1
        else:
            rperiod = np.random.rand(nperiod[0] + 1, nperiod[1] + 1)
        angles = 2 * np.pi * rperiod
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[: -d[0], : -d[1]]
        g10 = gradients[d[0] :, : -d[1]]
        g01 = gradients[: -d[0], d[1] :]
        g11 = gradients[d[0] :, d[1] :]

        # resolve size inconsistentcy isssue
        g_ysize, g_xsize = g00.shape[:2]
        grid_new = np.zeros((g_ysize, g_xsize, grid.shape[2]))
        for i in range(grid.shape[2]):
            grid_new[:, :, i] = cv2.resize(grid[:, :, i], (g_xsize, g_ysize), interpolation=cv2.INTER_AREA)

        # Ramps
        n00 = np.sum(np.dstack((grid_new[:, :, 0], grid_new[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid_new[:, :, 0] - 1, grid_new[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid_new[:, :, 0], grid_new[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid_new[:, :, 0] - 1, grid_new[:, :, 1] - 1)) * g11, 2)

        # Interpolation
        t = interpolant(grid_new)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        out = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        return cv2.resize(out, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)

    def generate_perlin_noise(
        self,
        input_size,
        nperiod,
        interpolant,
        octaves=1,
        persistence=0.5,
        lacunarity=2,
    ):
        """
        Main function to generate noise
        """

        noise = np.zeros(input_size)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            nperiods = (int(frequency * nperiod[0]), int(frequency * nperiod[1]))
            out = self.perlin_noise_2d(input_size, nperiods, interpolant)
            noise += amplitude * out
            frequency *= lacunarity
            amplitude *= persistence
        return noise

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

        # create mask of noises
        shape = (ysize, xsize)

        # get mask of noise and resize it to image size
        mask = self.generate_perlin_noise(
            shape,
            nperiod=self.nperiod,
            interpolant=self.interpolant,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
        )

        # adjust mask location for noise type = 1
        if self.noise_type == 1:
            mask_ysize, mask_xsize = mask.shape[:2]
            mask = mask[int(mask_ysize / 4) : int(mask_ysize / 4) * 3, :]

        # randomly rotate mask
        mask = np.rot90(mask, random.randint(0, 3))

        # new size after rotation
        mask_ysize, mask_xsize = mask.shape[:2]

        # rescale to 0 -255
        mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255
        if self.intensity[0] > self.intensity[1]:
            self.intensity[0] = self.intensity[1]

        # creates random small dot of noises
        if self.noise_type == 4:
            for _ in range(random.randint(10, 30)):
                region_size_x = random.randint(2, 6)
                region_size_y = random.randint(2, 6)
                offset = 12
                half_offset = int(offset / 2)
                # y start and y end
                ys = random.randint(0 + offset, mask_ysize - region_size_y - 1 - offset)
                ye = ys + region_size_y
                # x start and x end
                xs = random.randint(0 + offset, mask_xsize - region_size_x - 1 - offset)
                xe = xs + region_size_x
                mask[ys:ye, xs:xe] = (mask[ys:ye, xs:xe] / 10) - random.randint(self.intensity[0], self.intensity[1])
                larger_region = mask[ys - half_offset : ye + half_offset, xs - half_offset : xe + half_offset]
                larger_region = cv2.GaussianBlur(larger_region, (21, 21), cv2.BORDER_DEFAULT)
            mask = mask.astype("int") + random.randint(self.intensity[0], self.intensity[1])
            mask[mask < 0] = 0
            mask += random.randint(30, 120)
            mask[mask > 255] = 255
            mask = mask.astype("uint8")
        else:
            mask += random.randint(self.intensity[0], self.intensity[1])
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
        image_copy = image.copy()
        result_new = result.astype("int") + image_sobel.astype("int")
        image[image > result_new] = 0
        result_new[result_new > 255] = 0
        result_new[result_new > image_copy] = 0
        result = image + result_new

        return result.astype("uint8")

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            result = self.apply_augmentation(image)
            return result

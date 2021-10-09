import random

import cv2
import numpy as np

from augraphy.augmentations.lib import smooth
from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation


class BadPhotoCopy(Augmentation):
    """Uses Perlin noise to generate an effect of dirty copier.

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
        nperiod=(4, 4),
        octaves=4,
        persistence=0.5,
        lacunarity=2,
        wave_pattern=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.nperiod = nperiod
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.wave_pattern = wave_pattern

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BadPhotoCopy(nperiod={self.nperiod}, octaves={self.octaves}, persistence={self.persistence}, lacunarity={self.lacunarity}, wave_pattern={self.wave_pattern}, p={self.p})"

    def apply_wave(self, mask):
        """
        applies wavy pattern mask to input mask
        """

        # rescale mask from 0 to 255
        mask_rescaled = mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255).astype("uint8")
        mask_ysize, mask_xsize = mask_rescaled.shape
        img_wave = np.zeros_like(mask_rescaled)

        # get mask size measurements
        mask_y_one_quarter_size = int(mask_ysize / 4)
        mask_y_third_quarter_size = int(mask_ysize * 3 / 4)
        mask_y_one_twelve_size = int(mask_ysize / 12)
        mask_x_one_twelve_size = int(mask_xsize / 12)

        # generate points and at least 12 points
        number_points = random.randint(12, 18)

        # generate random x location
        x_points = [
            random.randint(mask_x_one_twelve_size, mask_xsize - mask_x_one_twelve_size)
            for _ in range(number_points - 2)
        ]
        x_points.sort()

        # 1st point
        points = [(0, random.randint(mask_y_one_quarter_size, mask_y_third_quarter_size))]
        for i in range(1, number_points - 1, 1):
            # points between 1st and last point
            points.append(
                (x_points[i - 1], random.randint(mask_y_one_quarter_size, mask_y_third_quarter_size)),
            )  # last point
        # last point
        points.append(
            (mask_xsize - 1, random.randint(mask_y_one_quarter_size, mask_y_third_quarter_size)),
        )  # last point

        # smooth points
        smooth_points = smooth(points, 10)

        for (x_point, y_point) in smooth_points:
            img_wave[: int(y_point), int(x_point)] = 1

            # additional noise, to smooth the edges between wavy mask
            p = random.randint(5, 15) / 10
            for y in range(int(y_point), int(mask_ysize - mask_y_one_twelve_size), 1):
                if p > 1:
                    break
                if random.random() > p:
                    img_wave[y, int(x_point)] = 1
                p += 0.005

        # get percentage of dark
        top_left = np.sum(mask_rescaled[:10, :10]) / (100 * 255)
        top_right = np.sum(mask_rescaled[:10, mask_xsize - 10 : mask_xsize]) / (100 * 255)
        bottom_left = np.sum(mask_rescaled[mask_ysize - 10 : mask_ysize : 10, :10]) / (100 * 255)
        bottom_right = np.sum(mask_rescaled[mask_ysize - 10 : mask_ysize, mask_xsize - 10 : mask_xsize]) / (100 * 255)

        # top
        if top_left < 0.3 and top_right < 0.3:
            mask = img_wave * mask
        # right
        elif top_right < 0.3 and bottom_right < 0.3:
            img_wave = np.rot90(img_wave, 3)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # bottom
        elif bottom_left < 0.3 and bottom_right < 0.3:
            img_wave = np.flipud(img_wave)
            mask = img_wave * mask
        # left
        elif top_left < 0.3 and bottom_left < 0.3:
            img_wave = np.rot90(img_wave, 1)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # topleft
        elif top_left < 0.3:
            mask = img_wave * mask
        # topright
        elif top_right < 0.3:
            img_wave = np.rot90(img_wave, 3)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # bottomleft
        elif bottom_left < 0.3:
            img_wave = np.rot90(img_wave, 1)
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask
        # bottomright
        elif bottom_right < 0.3:
            img_wave = np.flipud(img_wave)
            mask = img_wave * mask
        # other
        else:
            img_wave = np.rot90(img_wave, random.randint(0, 3))
            img_wave = cv2.resize(img_wave, (mask_xsize, mask_ysize), interpolation=cv2.INTER_AREA)
            mask = img_wave * mask

        mask[img_wave == 0] = 255

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
        angles = 2 * np.pi * np.random.rand(nperiod[0] + 1, nperiod[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[: -d[0], : -d[1]]
        g10 = gradients[d[0] :, : -d[1]]
        g01 = gradients[: -d[0], d[1] :]
        g11 = gradients[d[0] :, d[1] :]

        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

        # Interpolation
        t = interpolant(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        out = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        return cv2.resize(out, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)

    def generate_perlin_noise(
        self,
        input_size,
        nperiod,
        interpolant,
        octaves=4,
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

        # initialize vectorized mapping functions
        # add random zero value noise
        add_noise_fn = lambda x: 0 if (x <= random.randint(0, 224)) else 255
        add_noise = np.vectorize(add_noise_fn)

        # remove noise by using min
        remove_noise_fn = (
            lambda original, noised, mask: min(original, noised + mask)
            if (np.uint(noised) + np.uint(mask) <= 255)
            else original
        )
        remove_noise = np.vectorize(remove_noise_fn)

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

        mask += random.randint(30, 60)
        mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255).astype("uint8")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # apply mask
        if self.wave_pattern:
            mask = self.apply_wave(mask)

        # random flip mask vertically or horizontally
        if random.choice([True, False]):
            mask = cv2.flip(mask, 0)
        if random.choice([True, False]):
            mask = cv2.flip(mask, 1)

        # add noise and blur
        noise_img = add_noise(mask).astype("uint8")
        gaussian_kernel = (random.choice([3, 5, 7]), random.choice([3, 5, 7]))
        blurred = cv2.GaussianBlur(noise_img, gaussian_kernel, 0)

        # randomize noise type
        noise_img_type = random.randint(0, 2)

        # type 0 and type 1 noise
        if noise_img_type != 2:

            _, thresh1 = cv2.threshold(
                blurred,
                random.randint(64, 128),
                255,
                cv2.THRESH_BINARY,
            )

            # type 0 noise
            if noise_img_type == 0:
                result = cv2.multiply(thresh1, image, scale=1 / 255)
                result = remove_noise(image, result, image_sobel)

            # type 1 noise
            else:
                grey_img = noise_img.copy()
                grey_img[:, :][grey_img[:, :] == 0] = random.choice(
                    [255, 255, 255, random.randint(196, 224)],
                )

                noise_img = cv2.multiply(grey_img, thresh1, scale=1 / 255)
                result = cv2.multiply(noise_img, image, scale=1 / 255)
                result = remove_noise(image, result, image_sobel)

        # type 2 noise
        else:

            noise_img = cv2.multiply(noise_img, blurred, scale=1 / 255)
            result = cv2.multiply(noise_img, image, scale=1 / 255)
            result = remove_noise(image, result, image_sobel)

        return result

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            result = self.apply_augmentation(image)
            return result

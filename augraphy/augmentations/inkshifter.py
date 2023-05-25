"""
version: 0.0.1


Dependencies
*******************************************************************************
    - opencv
    - numpy


Documentation
********************************************************************************

    - Handwriter Repository: https://github.com/sherlockdoyle/Handwriter/tree/main
    - Noise Generation: https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
    - OpenCV remap() function : https://docs.opencv.org/3.4/d1/da0/tutorial_remap.html
    - Opencv meshgrid() function: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

"""
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class InkShifter(Augmentation):
    def __init__(
        self,
        text_shift_scale_range=(18, 27),
        text_shift_factor_range=(1, 4),
        text_fade_range=(0, 2),
        blur_kernel_size=(5, 5),
        blur_sigma=0,
        noise_type="random",
        p=1.0,
    ):
        """
        InkShifter augmentation shifts and displaces the image using noise maps.

        :param text_shift_scale_range (tuple): Range for the text shift scale.
        :param: text_shift_factor_range (tuple): Range for the text shift factor.
        :param: text_fade_range (tuple): Range for the text fade.
        :param: noise_type (str): Type of noise to use ("random", "perlin", or None).
        :param p (float): Probability of applying the augmentation.

        """
        super().__init__(p=p)
        self.text_shift_scale_range = text_shift_scale_range
        self.text_shift_factor_range = text_shift_factor_range
        self.text_fade_range = text_fade_range
        self.noise_type = noise_type
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def __repr(self):
        return f"InkShifter: text_shift_scale_range = {self.text_shift_scale_range}, text_shift_factor_range = {self.text_shift_factor_range}, text_fade_range = {self.text_fade_range}, noise_type = {self.noise_type}, blur_kernel_size = {self.blur_kernel_size}, blur_sigma = {self.blur_sigma}"

    def displace_image(self, img, mapx, mapy, fill=(255, 255, 255)):
        """
        Apply displacement map to an image.

        :param img: Input Image
        :param mapx (numpy.ndarray): x-componet of the displacement map
        :param mapy (numpy.ndarray): y component of the displacement map
        :param fill: Fill value of the pixels outside the image

        """
        gridx, gridy = np.meshgrid(
            np.arange(img.shape[1], dtype=np.float32),
            np.arange(img.shape[0], dtype=np.float32),
        )
        if mapx is None:
            mapx = gridx
        else:
            mapx += gridx
        if mapy is None:
            mapy = gridy
        else:
            mapy += gridy

        return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    def noise_map(self, shape, res=(64, 64)):
        """
        Generate a noise map based on Perlin Noise
        :param shape(tuple): Desired shape of the perlin noise map
        :param res(tuple): Resolution of the noise map

        """
        orig_shape = shape
        shape = np.ceil(shape[0] / res[0]) * res[0], np.ceil(shape[1] / res[1]) * res[1]

        d0, d1 = shape[0] // res[0], shape[1] // res[1]
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        grad = np.dstack((np.cos(angles), np.sin(angles)))
        grid = np.mgrid[: res[0] : res[0] / shape[0], : res[1] : res[1] / shape[1]].transpose(1, 2, 0) % 1
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * grad[:-1, :-1].repeat(d0, 0).repeat(d1, 1), 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * grad[1:, :-1].repeat(d0, 0).repeat(d1, 1), 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * grad[:-1, 1:].repeat(d0, 0).repeat(d1, 1), 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * grad[1:, 1:].repeat(d0, 0).repeat(d1, 1), 2)

        t = 6 * grid**5 - 15 * grid**4 + 10 * grid**3
        n0 = (1 - t[:, :, 0]) * n00 + t[:, :, 0] * n10
        n1 = (1 - t[:, :, 0]) * n01 + t[:, :, 0] * n11
        noise = (np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1))[: orig_shape[0], : orig_shape[1]].astype(
            np.float32,
        )
        noise_blurred = cv2.GaussianBlur(noise, self.blur_kernel_size, self.blur_sigma)
        return noise_blurred

    def noise_map_fractal(self, shape, res=(64, 64), octaves=1, persistence=0.5):
        """
        Generate a fractal noise map
        :param shape(tuple): desired shape of the fractal noise map
        :param res(tuple): resolution of the noise map
        :param octaves(int): Number of octaves in the fractal noise
        :param persistence (float): Persistence value for the fractal nois

        """
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.noise_map(shape, (frequency * res[0], frequency * res[1]))
            frequency *= 2
            amplitude *= persistence
        return noise.astype("float32")

    def put_fading(self, img, fade, f=0.5):
        """
        Apply fading effect to the image
        :param img(numpy.ndarray): input image
        :param fade(numpy.ndarray): fade values
        :param f(float): Fading factor

        """
        fade -= fade.min()
        fade /= fade.max()
        fade += (1 - fade) * f
        return (255 - (255 - img) * fade.reshape((fade.shape[0], fade.shape[1], 1))).astype(np.uint8)

    def __call__(self, image, layer=None, force=None):
        if force or self.should_run():
            h, w, _ = image.shape
            text_shift_scale = random.randint(self.text_shift_scale_range[0], self.text_shift_scale_range[1])
            text_shift_factor = random.randint(self.text_shift_factor_range[0], self.text_shift_factor_range[1])

            if self.noise_type == "random":
                perlin_noise = random.choice([True, False])
            elif self.noise_type == "perlin":
                perlin_noise = True
            if self.noise_type == "fractal":
                perlin_noise = False
            else:
                perlin_noise = True

            if perlin_noise:
                noisemap_x = self.noise_map((h, w), (text_shift_scale, text_shift_scale))
                noisemap_y = self.noise_map((h, w), (text_shift_scale, text_shift_scale))
                amp = random.random()
                disp_img = self.displace_image(
                    image,
                    -amp * text_shift_factor * noisemap_x,
                    text_shift_factor * noisemap_y,
                )

            else:
                noisemap_x = self.noise_map_fractal((h, w), (text_shift_scale, text_shift_scale))
                noisemap_y = self.noise_map_fractal((h, w), (text_shift_scale, text_shift_scale))
                amp = random.random()
                disp_img = self.displace_image(
                    image,
                    -amp * text_shift_factor * noisemap_x,
                    text_shift_factor * noisemap_y,
                )

            return disp_img

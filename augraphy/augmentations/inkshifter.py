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

from augraphy.augmentations.lib import update_mask_labels
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
        """InkShifter augmentation shifts and displaces the image using noise maps.

        :param text_shift_scale_range: Range for the text shift scale.
        :type text_shift_scale_range: tuple, optional
        :param: text_shift_factor_range: Range for the text shift factor.
        :type text_shift_factor_range: tuple, optional
        :param: text_fade_range: Range for the text fade.
        :type text_fade_range: tuple, optional
        :param: noise_type: Type of noise to use ("random", "perlin", or None).
        :type noise_type: string, optional
        :param p: The probability that this Augmentation will be applied.
        :type p: float, optional
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

    def displace_image(self, img, mask, mapx, mapy, fill=(255, 255, 255)):
        """Apply displacement map to an image.

        :param img: Input Image
        :type img: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param mapx: x-componet of the displacement map
        :type mapx: numpy array
        :param mapy: y component of the displacement map
        :type mapy: numpy array
        :param fill: Fill value of the pixels outside the image in BGR
        :type fill: tuple
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

        image_displaced = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

        if mask is not None:
            mask_labels = np.unique(mask).tolist() + [0]
            mask = cv2.remap(
                mask,
                mapx,
                mapy,
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill,
            )
            update_mask_labels(mask, mask_labels)

        return image_displaced, mask

    def noise_map(self, shape, res=(64, 64)):
        """Generate a noise map based on Perlin Noise

        :param shape: Desired shape of the perlin noise map
        :type shape: tuple
        :param res: Resolution of the noise map
        :type res: tuple, optional
        """

        orig_shape = shape
        shape = np.ceil(shape[0] / res[0]) * res[0], np.ceil(shape[1] / res[1]) * res[1]

        d0, d1 = shape[0] // res[0], shape[1] // res[1]
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        grad = np.dstack((np.cos(angles), np.sin(angles)))
        gysize, gxsize = grad.shape[:2]
        grid = np.mgrid[: res[0] : res[0] / shape[0], : res[1] : res[1] / shape[1]].transpose(1, 2, 0) % 1

        # grid y size is larger after the ceil rounding, prune it
        if grid.shape[0] > (gysize - 1) * d0:
            difference = int(abs(grid.shape[0] - (gysize - 1) * d0))
            grid = grid[:-difference, :]
        # grid y size is smaller after the ceil rounding, pad it
        elif grid.shape[0] < (gysize - 1) * d0:
            difference = int(abs(grid.shape[0] - (gysize - 1) * d0))
            grid = np.pad(
                grid,
                # (top, bottom), (left, right)
                pad_width=((0, difference), (0, 0), (0, 0)),
                mode="edge",
            )
        # grid x size is larger after the ceil rounding, prune it
        if grid.shape[1] > (gxsize - 1) * d1:
            difference = int(abs(grid.shape[1] - (gxsize - 1) * d1))
            grid = grid[:, :-difference]
        # grid x size is smaller after the ceil rounding, pad it
        elif grid.shape[1] < (gxsize - 1) * d1:
            difference = int(abs(grid.shape[1] - (gxsize - 1) * d1))
            grid = np.pad(
                grid,
                # (top, bottom), (left, right)
                pad_width=((0, 0), (0, difference), (0, 0)),
                mode="edge",
            )

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
        """Generate a fractal noise map

        :param shape: desired shape of the fractal noise map
        :type shape: tuple
        :param res: resolution of the noise map
        :type res: tuple, optional
        :param octaves: Number of octaves in the fractal noise
        :type octaves: int, optional
        :param persistence: Persistence value for the fractal nois
        :type persistence: float, optional
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
        """Apply fading effect to the image

        :param img: input image
        :type img: numpy array
        :param fade(numpy.ndarray): fade values
        :type face: numpy array
        :param f: Fading factor
        :type f: float, optional
        """

        fade -= fade.min()
        fade /= fade.max()
        fade += (1 - fade) * f
        return (255 - (255 - img) * fade.reshape((fade.shape[0], fade.shape[1], 1))).astype(np.uint8)

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            h, w = image.shape[:2]

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
            else:
                noisemap_x = self.noise_map_fractal((h, w), (text_shift_scale, text_shift_scale))
                noisemap_y = self.noise_map_fractal((h, w), (text_shift_scale, text_shift_scale))

            amp = random.random()
            image_displaced, mask = self.displace_image(
                image,
                mask,
                -amp * text_shift_factor * noisemap_x,
                text_shift_factor * noisemap_y,
            )

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_displaced] + outputs_extra
            else:
                return image_displaced

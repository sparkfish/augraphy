"""
version: 0.0.1
****************************


Dependencies:
****************************


Documentation:
****************************


"""
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation

# from augraphy.augmentations.shadowcast import ShadowCast

# from collections import namedtuple


class InkShifter(Augmentation):
    def __init__(
        self,
        text_shift_scale_range=(50, 60),
        text_shift_factor_range=(3, 5),
        text_fade_range=(0, 2),
        p=1.0,
    ):
        """ """
        super().__init__(p=p)
        self.text_shift_scale_range = text_shift_scale_range
        self.text_shift_factor_range = text_shift_factor_range
        self.text_fade_range = text_fade_range

    def displace_image(self, img, mapx, mapy, fill=(255, 255, 255)):
        """Apply displacement map to an image."""
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
        """Generate a noise map"""
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
        return (np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1))[: orig_shape[0], : orig_shape[1]].astype(
            np.float32,
        )

    def put_fading(self, img, fade, f=0.5):
        fade -= fade.min()
        fade /= fade.max()
        # fade = 1-(1-fade)**2
        fade += (1 - fade) * f
        return (255 - (255 - img) * fade.reshape((fade.shape[0], fade.shape[1], 1))).astype(np.uint8)

    def __call__(self, image, layer=None, force=None):
        if force or self.should_run():
            h, w, _ = image.shape
            # print("The shape of the image is", image.shape)
            # print("Text shift scale is", self.text_shift_scale_range)
            # print("Text shift factor is", self.text_shift_factor_range)
            text_shift_scale = random.randint(self.text_shift_scale_range[0], self.text_shift_scale_range[1])
            text_shift_factor = random.randint(self.text_shift_factor_range[0], self.text_shift_factor_range[1])
            # print("[INFO] The text shift scale is:", text_shift_scale)
            # print("[INFO] The text shift factor is:", text_shift_factor )
            noisemap_x = self.noise_map((h, w), (text_shift_scale, text_shift_scale))
            noisemap_y = self.noise_map((h, w), (text_shift_scale, text_shift_scale))
            amp = random.random()
            # print("[INFO] The amplification factor is", amp)
            disp_img = self.displace_image(
                image,
                -amp * text_shift_factor * noisemap_x,
                text_shift_factor * noisemap_y,
            )
            # background  = np.full((h,w,3), 250,  dtype = np.uint8)
            # # background = background.transpose(0,2,1)
            # print("The shape of the background is ", background.shape)
            # obj = ShadowCast()
            # img_shadowcast = obj(background)
            # text_shift_factor = 5.5
            # text_fade_factor = 1
            # faded_img = self.put_fading(disp_img, self.noise_map((h, w), (text_shift_scale, text_shift_scale)), text_fade_factor)
            # norm_back = cv2.normalize(
            # cv2.cvtColor(background, cv2.COLOR_BGR2GRAY),
            # None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            # )
            # page_morphed_img = self.displace_image(faded_img, None, 40-60*norm_back)
            # print(page_morphed_img)
            # cv2.imshow("background", norm_back)
            # cv2.waitKey(0)
            # on_page_img = cv2.normalize(
            # (background * (page_morphed_img) * 255).astype(np.uint8),
            # None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            # )

            # return disp_img
            return disp_img

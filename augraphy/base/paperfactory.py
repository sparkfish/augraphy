import glob
import os
import random

import cv2
import numpy as np

from augraphy.augmentations.brightness import Brightness
from augraphy.augmentations.lib import generate_average_intensity
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class PaperFactory(Augmentation):
    """Replaces the starting paper image with a texture randomly chosen from
    a directory and resized to fit or cropped and tiled to fit.

    :param texture_path: Directory location to pull paper textures from.
    :type texture_path: string, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        texture_path="./paper_textures",
        color_augmentation=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.texture_path = texture_path
        self.texture_file_names = []
        self.texture_file_name = None
        self.paper_textures = list()
        for file in glob.glob(f"{texture_path}/*"):
            texture = cv2.imread(file)
            self.texture_file_names.append(os.path.basename(file))
            # prevent invalid image file
            if hasattr(texture, "dtype") and texture.dtype == np.uint8:

                if len(texture.shape) > 2 and texture.shape[2] == 4:
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2BGR)
                elif len(texture.shape) > 2 and texture.shape[2] == 3:
                    pass
                else:
                    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

                self.paper_textures.append(cv2.imread(file))

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"PaperFactory(texture_path={self.texture_path}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            if self.paper_textures:
                shape = image.shape
                random_index = random.randint(0, len(self.paper_textures) - 1)
                texture = self.paper_textures[random_index]
                self.texture_file_name = self.texture_file_names[random_index]
                # reset file names and textures
                self.texture_file_names = []
                self.paper_textures = []

                # If the texture we chose is larger than the paper,
                # get randomn location that fit into paper size
                if (texture.shape[0] >= shape[0]) and (texture.shape[1] >= shape[1]):
                    difference_y = texture.shape[0] - shape[0]
                    difference_x = texture.shape[1] - shape[1]
                    start_y = random.randint(0, difference_y - 1)
                    start_x = random.randint(0, difference_x - 1)
                    texture = texture[start_y : start_y + shape[0], start_x : start_x + shape[1]]

                # If the texture we chose is smaller in either dimension than the paper,
                # use the resize logic
                else:
                    texture = self.resize(texture, shape)

                # texture_intensity
                texture_intensity = generate_average_intensity(texture)
                # brighten dark texture based on target intensity, max intensity = 255 (brightest)
                target_intensity = 200
                if texture_intensity < target_intensity:
                    brighten_ratio = abs(texture_intensity - target_intensity) / texture_intensity
                    brighten_min = 1 + (brighten_ratio / 2)
                    brighten_max = 1 + brighten_ratio
                    brightness = Brightness(range=(brighten_min, brighten_max), min_brightness=1)
                    texture = brightness(texture)
                return texture

            else:
                print("No paper image in the paper directory!")

    # Scales and zooms a given texture to fit a given shape.
    def resize(self, texture, shape):
        texture_h = texture.shape[0]
        texture_w = texture.shape[1]
        shape_h = shape[0]
        shape_w = shape[1]

        if texture_h > shape_h or texture_w > shape_w:  # Zoom out
            h_ratio = shape_h / texture_h
            w_ratio = shape_w / texture_w

            if h_ratio > w_ratio:
                scale = random.uniform(h_ratio, 1.2)
            else:
                scale = random.uniform(w_ratio, 1.2)

            zoom = (int(texture_w * scale), int(texture_h * scale))
            # print(f"Zoom out from {texture.shape} to {zoom}")
            texture = cv2.resize(texture, zoom)
            texture_h = texture.shape[0]
            texture_w = texture.shape[1]

        if texture_h <= shape_h or texture_w <= shape_w:  # Zoom in
            h_ratio = shape_h / texture_h
            w_ratio = shape_w / texture_w

            if h_ratio > w_ratio:
                scale = random.uniform(h_ratio, h_ratio + 1.5)
            else:
                scale = random.uniform(w_ratio, w_ratio + 1.5)
            zoom = (int(texture_w * scale), int(texture_h * scale))
            # print(f"Zoom in from {texture.shape} to {zoom}")
            texture = cv2.resize(texture, zoom)

        return texture

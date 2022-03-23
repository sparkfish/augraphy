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

    :param tile_texture_shape: Pair of ints determining the range from which to
           sample the texture dimensions.
    :type tile_texture_shape: tuple, optional
    :param texture_path: Directory location to pull paper textures from.
    :type texture_path: string, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        tile_texture_shape=(250, 250),
        texture_path="./paper_textures",
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.paper_textures = list()
        self.tile_texture_shape = tile_texture_shape
        self.texture_path = texture_path
        self.texture_file_names = []
        self.texture_file_name = None
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
        return (
            f"PaperFactory(tile_texture_shape={self.tile_texture_shape}, texture_path={self.texture_path}, p={self.p})"
        )

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            if self.paper_textures:
                shape = image.shape

                random_index = random.randint(0, len(self.paper_textures) - 1)
                texture = self.paper_textures[random_index]

                # If the texture we chose is larger than the paper,
                # just align to the top left corner and crop as necessary
                if (texture.shape[0] >= shape[0]) and (texture.shape[1] >= shape[1]):
                    texture = texture[0 : shape[0], 0 : shape[1]]
                    return texture

                # If the texture we chose is smaller in either dimension than the paper,
                # use the resize logic
                else:
                    texture = self.resize(texture, shape)
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

    # Returns a paper texture cropped to a given shape.
    def get_texture(self, shape):
        random_index = random.randint(0, len(self.paper_textures) - 1)
        texture = self.paper_textures[random_index]
        self.texture_file_name = self.texture_file_names[random_index]
        # reset file names and textures
        self.texture_file_names = []
        self.paper_textures = []

        # texture_intensity
        texture_intensity = generate_average_intensity(texture)
        # brighten dark texture based on target intensity, max intensity = 255 (brightest)
        target_intensity = 200
        if texture_intensity < target_intensity:
            brighten_ratio = abs(texture_intensity - target_intensity) / texture_intensity
            brighten_min = 1 + brighten_ratio
            brighten_max = 1 + brighten_ratio + 0.5
            brightness = Brightness(range=(brighten_min, brighten_max))
            texture = brightness(texture)

        if texture.shape[0] < shape[0] or texture.shape[1] < shape[1]:
            texture = self.resize(texture, shape)

        h = random.randint(0, texture.shape[0] - shape[0])
        w = random.randint(0, texture.shape[1] - shape[1])
        cropped_texture = texture[h : h + shape[0], w : w + shape[1]]
        return cropped_texture

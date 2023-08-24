import glob
import os
import random

import cv2
import numpy as np

from augraphy.augmentations.brightness import Brightness
from augraphy.augmentations.lib import generate_average_intensity
from augraphy.augmentations.lib import generate_texture
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.utilities.overlaybuilder import OverlayBuilder


class PaperFactory(Augmentation):
    """Replaces the starting paper image with a texture randomly chosen from
    a directory and resized to fit or cropped and tiled to fit.

    :param texture_path: Directory location to pull paper textures from.
    :type texture_path: string, optional
    :param blend_texture: Flag to blend multiple textures.
    :type blend_texture: int, optional
    :param blend_method: The method to blend multiple textures.
    :type blend_method: string, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        texture_path="./paper_textures",
        blend_texture=1,
        blend_method="random",
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.texture_path = texture_path
        self.blend_texture = blend_texture
        self.blend_method = blend_method
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
        return f"PaperFactory(texture_path={self.texture_path},blend_texture={self.blend_texture}, blend_method={self.blend_method}, p={self.p})"

    def retrieve_texture(self, image):
        shape = image.shape
        random_index = random.randint(0, len(self.paper_textures) - 1)
        texture = self.paper_textures[random_index]
        texture = np.rot90(texture, random.randint(1, 4))
        self.texture_file_name = self.texture_file_names[random_index]
        # reset file names and textures
        """
        self.texture_file_names = []
        self.paper_textures = []
        """

        # check for edge
        texture = self.check_paper_edges(texture)

        # If the texture we chose is larger than the paper,
        # get random location that fit into paper size
        if (texture.shape[0] > shape[0]) and (texture.shape[1] > shape[1]):
            difference_y = texture.shape[0] - shape[0]
            difference_x = texture.shape[1] - shape[1]
            start_y = random.randint(0, difference_y)
            start_x = random.randint(0, difference_x)
            texture = texture[start_y : start_y + shape[0], start_x : start_x + shape[1]]

        # If the texture we chose is smaller in either dimension than the paper,
        # use the resize logic
        else:
            texture = self.resize(texture, shape)

        return texture

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            # get texture from paper
            if self.paper_textures:
                # get image texture
                texture = self.retrieve_texture(image)
            # generate random mask as texture
            else:
                ysize, xsize = image.shape[:2]
                sigma = random.uniform(3, 5)
                turbulence = random.randint(3, 9)
                texture = generate_texture(ysize, xsize, channel=1, value=255, sigma=sigma, turbulence=turbulence)

                if len(image.shape) > 2:
                    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

            # blend multiple textures
            if self.blend_texture:
                if len(texture.shape) < 3:
                    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

                # get another image as texture
                if len(self.paper_textures) > 0:
                    new_texture = self.retrieve_texture(image)
                    if len(new_texture.shape) < 3:
                        new_texture = cv2.cvtColor(new_texture, cv2.COLOR_GRAY2BGR)

                # generate noise as texture if no texture is provided
                else:
                    ysize, xsize = texture.shape[:2]
                    sigma = random.uniform(3, 5)
                    turbulence = random.randint(3, 9)
                    new_texture = generate_texture(
                        ysize,
                        xsize,
                        channel=1,
                        value=255,
                        sigma=sigma,
                        turbulence=turbulence,
                    )
                    new_texture = cv2.cvtColor(new_texture, cv2.COLOR_GRAY2BGR)

                # resize for size consistency between both textures
                new_texture = cv2.resize(
                    new_texture,
                    (texture.shape[1], texture.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

                if self.blend_method == "random":
                    blend_method = random.choice(
                        [
                            "ink_to_paper",
                            "min",
                            "max",
                            "mix",
                            "normal",
                            "lighten",
                            "darken",
                            "screen",
                            "dodge",
                            "multiply",
                            "divide",
                            "grain_merge",
                            "overlay",
                            "FFT",
                        ],
                    )
                else:
                    blend_method = self.blend_method

                # Create overlay object and blend textures
                ob = OverlayBuilder(
                    blend_method,
                    new_texture,
                    texture,
                    1,
                    (1, 1),
                    "center",
                    0,
                )

                texture = ob.build_overlay()

            # texture_intensity
            texture_intensity = generate_average_intensity(texture)
            # brighten dark texture based on target intensity, max intensity = 255 (brightest)
            target_intensity = 200
            if texture_intensity < target_intensity:
                brighten_ratio = abs(texture_intensity - target_intensity) / texture_intensity
                brighten_min = 1 + (brighten_ratio / 2)
                brighten_max = 1 + brighten_ratio
                brightness = Brightness(brightness_range=(brighten_min, brighten_max), min_brightness=1)
                texture = brightness(texture)
            return texture

    def check_paper_edges(self, texture):
        """Crop image section with better texture.

        :param texture: Texture image.
        :type texture: numpy array
        """

        ysize, xsize = texture.shape[:2]

        # get single channel image
        if len(texture.shape) > 2:
            texture_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        else:
            texture_gray = texture.copy()

        # blur image
        texture_blur = cv2.GaussianBlur(texture_gray, (5, 5), 0)

        # convert to binary using otsu
        _, texture_binary = cv2.threshold(texture_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # get border average intensity
        border_average = (
            np.average(texture_binary[:, :10])
            + np.average(texture_binary[:, -10:])
            + np.average(texture_binary[:10, :])
            + np.average(texture_binary[-10:, :])
        ) / 4

        # get center average intensity
        center_x = int(xsize / 2)
        center_y = int(ysize / 2)
        center_average = np.average(texture_blur[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10])

        # if border intensity is higher, complement image
        if border_average > center_average:
            texture_binary = 255 - texture_binary

        # erode
        texture_binary = cv2.erode(texture_binary, np.ones((9, 9), np.uint8), iterations=1)

        # find contours in image
        contours, hierarchy = cv2.findContours(texture_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get all areas
        areas = [cv2.contourArea(contour) for contour in contours]

        # sort area and get the largest contour first
        area_indexs = list(np.argsort(areas))
        area_indexs.reverse()

        # threshold for contour
        min_area = ysize * xsize * 0.65

        # at least 1 contour
        if len(areas) > 0:
            max_contour = contours[area_indexs[0]]
            for i, area_index in enumerate(area_indexs):
                # last index
                if i == len(area_indexs) - 1:
                    # current contour >= min area
                    if areas[area_indexs[i]] >= min_area:
                        max_contour = contours[area_index]
                    else:
                        return texture
                else:
                    # current contour >= min area but the next one < min area
                    if areas[area_indexs[i]] >= min_area and areas[area_indexs[i + 1]] < min_area:
                        max_contour = contours[area_index]
                        break
                    else:
                        return texture

            # get rotated rectangle and their box
            rectangle = cv2.minAreaRect(max_contour)
            bbox = np.int0(cv2.boxPoints(rectangle))

            # get min of x and y from rectangle
            y_list = np.sort(bbox[:, 1])
            x_list = np.sort(bbox[:, 0])
            y_top = y_list[1]
            y_bottom = y_list[2]
            x_left = x_list[1]
            x_right = x_list[2]

            # crop texture
            texture_cropped = texture[y_top:y_bottom, x_left:x_right]
        else:
            texture_cropped = texture

        return texture_cropped

    def resize(self, texture, shape):
        """Scales and zooms a given texture to fit a given shape.

        :param texture: Texture image.
        :type texture: numpy array.3.
        :param shape: x and y shape of scaled image.
        :type shape: list or tuple
        """

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

import os
import random
import shutil
from glob import glob

import cv2
import matplotlib
import numpy as np
import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from augraphy.augmentations.lib import add_noise
from augraphy.augmentations.lib import binary_threshold
from augraphy.augmentations.lib import rotate_image
from augraphy.base.augmentation import Augmentation


class PencilScribbles(Augmentation):
    """Applies pencil scribbles to image.

    :param scribbles_type: Types of scribbles, choose from "random", "lines" or "text".
    :type scribbles_type: string, optional
    :param scribbles_location: Tuple of ints or floats (x,y) determining location of scribbles effect
           or use "random" for random location.
           The value will be in percentage of the image size if the value is float and in between 0 - 1:
           x (int) = image width  * x (float and 0 - 1);
           y (int) = image height * y (float and 0 - 1)
    :type scribbles_location: tuple or string, optional
    :param scribbles_size_range: Pair of floats determining the range for
           the size of the scribble to be created.
    :type scribbles_size_range: tuple, optional
    :param scribbles_count_range: Pair of floats determining the range for
           the number of scribbles to create.
    :type scribbles_count_range: tuple, optional
    :param scribbles_thickness_range: Pair of floats determining the range for
           the size of the scribbles to create.
    :type scribbles_thickness_range: tuple, optional
    :param scribbles_brightness_change: A list of value change for the brightness of
           the strokes. Default 128 creates a graphite-like appearance.
           32 creates a charcoal-like appearance.
           If more than one value is provided, the final value will be randomly selected.
    :type scribbles_brightness_change: list, optional
    :param scribbles_text: Text value for "text" based scribbles.
    :type scribbles_text: string, optional
    :param scribbles_text_font: Font types for "text" based scribbles.
            It can be the path to the ttf file, a path to the folder contains ttf files,
            an url to the ttf file, or simply "random" to use default randomized font types.
    :type scribbles_text_font: string, optional
    :param scribbles_text_rotate_range: Tuple of ints to determine rotation angle of "text" based scribbles.
    :type scribbles_text_rotate_range: tuple, optional
    :param scribbles_lines_stroke_count_range: Pair of floats determining the range for
           the number of strokes to create in each scribble.
    :type scribbles_lines_stroke_count_range: tuple, optional

    :param p: Probability of this Augmentation being applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scribbles_type="random",
        scribbles_location="random",
        scribbles_size_range=(250, 600),
        scribbles_count_range=(1, 6),
        scribbles_thickness_range=(1, 3),
        scribbles_brightness_change=[32, 64, 128],
        scribbles_text="random",
        scribbles_text_font="random",
        scribbles_text_rotate_range=(0, 360),
        scribbles_lines_stroke_count_range=(1, 6),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.scribbles_type = scribbles_type
        self.scribbles_location = scribbles_location
        self.scribbles_size_range = scribbles_size_range
        self.scribbles_count_range = scribbles_count_range
        self.scribbles_thickness_range = scribbles_thickness_range
        self.scribbles_brightness_change = scribbles_brightness_change
        self.scribbles_text = scribbles_text
        self.scribbles_text_font = scribbles_text_font
        self.scribbles_text_rotate_range = scribbles_text_rotate_range
        self.scribbles_lines_stroke_count_range = scribbles_lines_stroke_count_range
        self.fonts_directory = "fonts/"

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"PencilScribbles(scribbles_type={self.scribbles_type}, scribbles_location={self.scribbles_location}, scribbles_size_range={self.scribbles_size_range}, scribbles_count_range={self.scribbles_count_range}, scribbles_thickness_range={self.scribbles_thickness_range}, scribbles_brightness_change={self.scribbles_brightness_change}, scribbles_text={self.scribbles_text}, scribbles_text_font={self.scribbles_text_font}, scribbles_text_rotate_range={self.scribbles_text_rotate_range}, scribbles_lines_stroke_count_range={self.scribbles_lines_stroke_count_range}, p={self.p})"

    def apply_pencil_scribbles(self, scribbles_image, image):
        """Apply image with pencil strokes to background image.

        :param scribbles_image: Image with pencil scribbles.
        :type scribbles_image: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        """
        scribbles_image = cv2.cvtColor(scribbles_image, cv2.COLOR_BGR2GRAY)
        noise_mask = add_noise(scribbles_image, (0.3, 0.5), (32, 128), 0)
        scribbles_image[scribbles_image < 64] = noise_mask[scribbles_image < 64]

        scribbles_image = cv2.cvtColor(scribbles_image, cv2.COLOR_GRAY2BGR)
        scribbles_image = cv2.GaussianBlur(scribbles_image, (3, 3), 0)

        hsv = cv2.cvtColor(scribbles_image.astype("uint8"), cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] += random.choice(self.scribbles_brightness_change)
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        scribbles_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return cv2.multiply(scribbles_image, image, scale=1 / 255)

    def create_scribble(self, max_height, max_width, scribbles_type):
        """Create  pencil scribbles effect in an image.

        :param max_height: Maximum height of scribble effect.
        :type max_height: int
        :param max_width: Maximum width of scribble effect.
        :type max_width: int
        :param scribbles_type: Types of scribbles effect.
        :type scribbles_type: string
        """

        # line based scribbles
        if scribbles_type == "lines":

            size = random.randint(max(self.scribbles_size_range[0], 30), max(40, self.scribbles_size_range[1]))
            size = min([size, max_height, max_width])
            width, height = size, size  # picture's size

            # create blank white image
            scribbles_image = np.full((height, width, 3), fill_value=255, dtype="uint8")

            # draw each stroke
            lines_stroke_count_range = random.randint(
                self.scribbles_lines_stroke_count_range[0],
                self.scribbles_lines_stroke_count_range[1],
            )
            for i in range(lines_stroke_count_range):
                # create image with single scribble
                scribble_image = np.full((height, width, 3), fill_value=255, dtype="uint8")

                x = np.array(
                    [
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                    ],
                )
                y = np.array(
                    [
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                        random.randint(5, size - 25),
                    ],
                )

                start_stop = [
                    random.randint(5, size // 2),
                    random.randint(size // 2, size - 5),
                ]

                # Initilaize y axis
                lspace = np.linspace(min(start_stop), max(start_stop))

                # calculate the coefficients.
                z = np.polyfit(x, y, 2)

                # calculate x axis
                line_fitx = z[0] * lspace**2 + z[1] * lspace + z[2]
                verts = np.array(list(zip(line_fitx.astype(int), lspace.astype(int))))
                cv2.polylines(
                    scribble_image,
                    [verts],
                    False,
                    (0, 0, 0),
                    thickness=random.randint(
                        self.scribbles_thickness_range[0],
                        self.scribbles_thickness_range[1],
                    ),
                )

                # apply pencil scribbles effect
                scribbles_image = self.apply_pencil_scribbles(scribble_image, scribbles_image)

        # text based scribbles
        else:

            # create blank white image
            scribbles_image = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # create image with single scribble
            scribble_image = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # convert image to PIL
            scribble_image_PIL = Image.fromarray(scribble_image)
            draw = ImageDraw.Draw(scribble_image_PIL)
            # set font and size
            fonts_list = glob(self.fonts_directory + "/*.ttf")
            font = ImageFont.truetype(
                random.choice(fonts_list),
                size=int(random.randint(self.scribbles_size_range[0], self.scribbles_size_range[1]) / 8),
            )
            if self.scribbles_text == "random":
                text = random.choice(["Demo", "Approved", "Checked", "Original", "Copy", "Confidential"])
            else:
                text = self.scribbles_text

            # draw text
            draw.text(
                (int(max_width / 2), int(max_height / 2)),
                text,
                font=font,
                stroke_width=random.randint(self.scribbles_thickness_range[0], self.scribbles_thickness_range[1]),
                fill=(0, 0, 0),
            )

            # convert it back to numpy array
            scribble_image = np.array(scribble_image_PIL)

            # rotate image
            scribble_image = rotate_image(
                scribble_image,
                random.randint(self.scribbles_text_rotate_range[0], self.scribbles_text_rotate_range[1]),
            )

            # resize to make sure rotated image size is consistent
            scribble_image = cv2.resize(scribble_image, (max_width, max_height), interpolation=cv2.INTER_AREA)

            # apply pencil scribbles effect
            scribbles_image = self.apply_pencil_scribbles(scribble_image, scribbles_image)

            # remove additional blank area
            binary_image = binary_threshold(scribbles_image, threshold_method="threshold_otsu", threshold_arguments={})
            coordinates = cv2.findNonZero(255 - binary_image)
            x, y, w, h = cv2.boundingRect(coordinates)
            scribbles_image = scribbles_image[y : y + h, x : x + w]

        return scribbles_image

    def paste_scribbles(self, paste, target):
        """Paste image to another image.

        :param paste: Image for the paste effect.
        :type paste: numpy.array (numpy.uint8)
        :param target: The image to be pasted.
        :type target: numpy.array (numpy.uint8)
        """

        target_shape_length = len(target.shape)

        # scribbles is always in 3 channels, we need to check and convert if target is not in 3 channels
        if target_shape_length < 3:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)

        tysize, txsize = target.shape[:2]
        pysize, pxsize = paste.shape[:2]

        if self.scribbles_location == "random":
            target_x = random.randint(0, txsize - pxsize)
            target_y = random.randint(0, tysize - pysize)
        else:
            target_x, target_y = self.scribbles_location

            # check if provided location is float and scale them with target size
            if target_x >= 0 and target_x <= 1 and isinstance(target_x, float):
                target_x = int(target_x * txsize)
            if target_y >= 0 and target_y <= 1 and isinstance(target_y, float):
                target_y = int(target_y * tysize)

            # check if the location is within the target image size
            if target_x < 0:
                target_x = 0
            elif target_x + pxsize >= txsize:
                target_x = txsize - pxsize - 1

            if target_y < 0:
                target_y = 0
            elif target_y + pysize >= tysize:
                target_y = tysize - pysize - 1

        target[
            target_y : target_y + paste.shape[0],
            target_x : target_x + paste.shape[1],
        ] = paste

        # convert target back to original channel
        if target_shape_length < 3:
            target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

        return target

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            scribbles_background = np.full(image.shape, 255).astype("uint8")

            if self.scribbles_type == "text" or self.scribbles_type == "random":
                # create fonts directory
                os.makedirs(self.fonts_directory, exist_ok=True)

                if self.scribbles_text_font != "random":

                    # Check if it is a path to ttf file
                    if os.path.isfile(self.scribbles_text_font):
                        if self.scribbles_text_font.endswith("ttf"):

                            # remove all existing file
                            shutil.rmtree(self.fonts_directory)
                            os.makedirs(self.fonts_directory, exist_ok=True)

                            # move the ttf file into fonts directory
                            shutil.copy(self.scribbles_text_font, self.fonts_directory)

                        # if the path is not valid, set to default random fonts
                        else:
                            print("Invalid font.ttf file!")
                            self.scribbles_text_font = "random"

                    # Check if it is a folder
                    elif os.path.isdir(self.scribbles_text_font):
                        file_list = glob(self.scribbles_text_font + "/*.ttf")
                        if len(file_list) > 0:
                            self.fonts_directory = self.scribbles_text_font
                        else:
                            print("No font.ttf file in the directory!")
                            self.scribbles_text_font = "random"

                    # Check if it is a valid url
                    else:
                        try:
                            # remove all existing file
                            shutil.rmtree(self.fonts_directory)
                            os.makedirs(self.fonts_directory, exist_ok=True)

                            # download new ttf file
                            response = requests.get(self.scribbles_text_font)
                            open("fonts/font_type.zip", "wb").write(response.content)
                            shutil.unpack_archive("fonts/font_type.zip", self.fonts_directory)
                        except Exception:
                            print("Font url is not valid")
                            self.scribbles_text_font = "random"

                # Download random fonts or get it from system fonts
                if self.scribbles_text_font == "random":

                    file_list = glob("fonts/*.ttf")
                    if len(file_list) < 1:

                        # source: https://www.fontsquirrel.com/fonts/list/tag/handwritten
                        urls = [
                            "https://www.fontsquirrel.com/fonts/download/Jinky",
                            "https://www.fontsquirrel.com/fonts/download/Journal",
                            "https://www.fontsquirrel.com/fonts/download/indie-flower",
                        ]

                        # choose random font
                        url = random.choice(urls)

                        # try to download from url first
                        try:
                            # download from url and unzip them into font directory
                            response = requests.get(url)
                            open("fonts/font_type.zip", "wb").write(response.content)
                            shutil.unpack_archive("fonts/font_type.zip", self.fonts_directory)

                        # get system font if download failed
                        except Exception:
                            # From here, looks like this is the only solution to get system fonts
                            # https://stackoverflow.com/questions/65141291/get-a-list-of-all-available-fonts-in-pil
                            system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

                            # move the ttf file into fonts directory
                            shutil.copy(np.random.choice(system_fonts), self.fonts_directory)

            # Iterations to apply scribbles into image
            scribbles_count_range = random.randint(self.scribbles_count_range[0], self.scribbles_count_range[1])
            for i in range(scribbles_count_range):

                if self.scribbles_type == "random":
                    scribbles_type = random.choice(["lines", "text"])
                else:
                    scribbles_type = self.scribbles_type

                scribbles_image = self.create_scribble(image.shape[0], image.shape[1], scribbles_type)

                scribbles_merged = self.paste_scribbles(scribbles_image, scribbles_background)

                image = cv2.multiply(scribbles_merged, image, scale=1 / 255)

            return image

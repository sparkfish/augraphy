import os
import random
import shutil
from glob import glob

import matplotlib
import numpy as np
import requests

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.inkgenerator import InkGenerator


class Scribbles(Augmentation):
    """Applies scribbles to image.

    :param scribbles_type: Types of scribbles, choose from "random", "lines" or "text".
    :type scribbles_type: string, optional
    :param scribbles_ink: Types of scribbles ink, choose from "random", "pencil", "pen" or "marker".
    :type scribbles_ink: string, optional
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
    :param scribbles_skeletonize: Flag to enable skeletonization effect.
    :type scribbles_skeletonize: int, optional
    :param scribbles_skeletonize_iterations: Tuple of ints determing number of skeletonization iterations.
    :type scribbles_skeletonize_iterations: tuple, optional

    :param scribbles_color: Tuple of ints (BGR) determining the color of scribbles, or use "random" for random color.
    :type scribbles_color: tuple, optional
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
        scribbles_ink="random",
        scribbles_location="random",
        scribbles_size_range=(400, 600),
        scribbles_count_range=(1, 6),
        scribbles_thickness_range=(1, 3),
        scribbles_brightness_change=[32, 64, 128],
        scribbles_skeletonize=0,
        scribbles_skeletonize_iterations=(2, 3),
        scribbles_color="random",
        scribbles_text="random",
        scribbles_text_font="random",
        scribbles_text_rotate_range=(0, 360),
        scribbles_lines_stroke_count_range=(1, 6),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.scribbles_type = scribbles_type
        self.scribbles_ink = scribbles_ink
        self.scribbles_location = scribbles_location
        self.scribbles_size_range = scribbles_size_range
        self.scribbles_count_range = scribbles_count_range
        self.scribbles_thickness_range = scribbles_thickness_range
        self.scribbles_brightness_change = scribbles_brightness_change
        self.scribbles_skeletonize = scribbles_skeletonize
        self.scribbles_skeletonize_iterations = scribbles_skeletonize_iterations
        self.scribbles_color = scribbles_color
        self.scribbles_text = scribbles_text
        self.scribbles_text_font = scribbles_text_font
        self.scribbles_text_rotate_range = scribbles_text_rotate_range
        self.scribbles_lines_stroke_count_range = scribbles_lines_stroke_count_range
        self.fonts_directory = "fonts/"

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"PencilScribbles(scribbles_type={self.scribbles_type}, scribbles_ink={self.scribbles_ink}, scribbles_location={self.scribbles_location}, scribbles_size_range={self.scribbles_size_range}, scribbles_count_range={self.scribbles_count_range}, scribbles_thickness_range={self.scribbles_thickness_range}, scribbles_brightness_change={self.scribbles_brightness_change}, scribbles_skeletonize={self.scribbles_skeletonize}, scribbles_skeletonize_iterations={self.scribbles_skeletonize_iterations}, scribbles_color={self.scribbles_color}, scribbles_text={self.scribbles_text}, scribbles_text_font={self.scribbles_text_font}, scribbles_text_rotate_range={self.scribbles_text_rotate_range}, scribbles_lines_stroke_count_range={self.scribbles_lines_stroke_count_range}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

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

            # initialize random parameters
            fonts_list = glob(self.fonts_directory + "/*.ttf")

            if self.scribbles_type == "random":
                scribbles_type = random.choice(["lines", "texts"])
            else:
                scribbles_type = self.scribbles_type

            if self.scribbles_ink == "random":
                scribbles_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
            else:
                scribbles_ink = self.scribbles_ink

            if self.scribbles_skeletonize == "random":
                scribbles_skeletonize = random.choice([0, 1])
            else:
                scribbles_skeletonize = self.scribbles_skeletonize

            if self.scribbles_color == "random":
                scribbles_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                scribbles_color = self.scribbles_color

            if self.scribbles_location != "random":
                ysize, xsize = image.shape[:2]
                target_x, target_y = self.scribbles_location
                # check if provided location is float and scale them with target size
                if target_x >= 0 and target_x <= 1 and isinstance(target_x, float):
                    target_x = int(target_x * xsize)
                if target_y >= 0 and target_y <= 1 and isinstance(target_y, float):
                    target_y = int(target_y * ysize)
                scribbles_location = (target_x, target_y)
            else:
                scribbles_location = self.scribbles_location

            # create an ink generator and generate scribbles
            ink_generator = InkGenerator(
                ink_type=scribbles_ink,
                ink_draw_method=scribbles_type,
                ink_draw_iterations=self.scribbles_count_range,
                ink_location=scribbles_location,
                ink_background=image,
                ink_background_size=None,
                ink_background_color=None,
                ink_color=scribbles_color,
                ink_draw_size_range=self.scribbles_size_range,
                ink_thickness_range=self.scribbles_thickness_range,
                ink_brightness_change=self.scribbles_brightness_change,
                ink_skeletonize=scribbles_skeletonize,
                ink_skeletonize_iterations_range=self.scribbles_skeletonize_iterations,
                ink_text=self.scribbles_text,
                ink_text_font=fonts_list,
                ink_text_rotate_range=self.scribbles_text_rotate_range,
                ink_lines_coordinates="random",
                ink_lines_stroke_count_range=self.scribbles_lines_stroke_count_range,
            )

            image_output = ink_generator.generate_ink()

            return image_output

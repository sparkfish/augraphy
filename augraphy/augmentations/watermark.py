import random

import cv2
import numpy as np

from augraphy.augmentations.lib import rotate_image
from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class WaterMark(Augmentation):
    """
    Add watermark effect into input image.

    :param watermark_word: Word for watermark effect.
    :type watermark_word: string, optional
    :param watermark_font_size: Pair of ints to determine font size of watermark effect.
    :type watermark_font_size: tuple, optional
    :param watermark_font_thickness: Pair of ints to determine thickness of watermark effect.
    :type watermark_font_thickness: tuple, optional
    :param watermark_font_type: Font type of watermark effect.
    :type watermark_font_type: cv2 font types, optional
    :param watermark_rotation: Pair of ints to determine angle of rotation in watermark effect.
    :type watermark_rotation: tuple, optional
    :param watermark_location: Location of watermark effect, select from top, bottom, left, right, center and random.
    :type watermark_location: string, optional
    :param watermark_color: Triplets of ints to determine RGB color of watermark effect.
    :type watermark_color: tuple, optional
    :param watermark_method: Method to overlay watermark foreground into input image.
    :type watermark_method: string, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        watermark_word="random",
        watermark_font_size=(10, 15),
        watermark_font_thickness=(20, 25),
        watermark_font_type=cv2.FONT_HERSHEY_SIMPLEX,
        watermark_rotation=(0, 360),
        watermark_location="random",
        watermark_color="random",
        watermark_method="darken",
        p=1,
    ):
        super().__init__(p=p)
        self.watermark_word = watermark_word
        self.watermark_font_size = watermark_font_size
        self.watermark_font_thickness = watermark_font_thickness
        self.watermark_font_type = watermark_font_type
        self.watermark_rotation = watermark_rotation
        self.watermark_location = watermark_location
        self.watermark_color = watermark_color
        self.watermark_method = watermark_method

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"WaterMark(watermark_word={self.watermark_word}, watermark_font_size={self.watermark_font_size}, watermark_font_thickness={self.watermark_font_thickness}, watermark_font_type={self.watermark_font_type}, watermark_rotation={self.watermark_rotation}, watermark_location={self.watermark_location}, watermark_color={self.watermark_color}, watermark_method={self.watermark_method}, p={self.p})"

    # Create watermark
    def create_watermark(self, image):

        # initialize watermark word
        if self.watermark_word == "random":
            watermark_word = random.choice(
                ["COPY", "VOID", "DRAFT", "CONFIDENTIAL", "UNOFFICIAL", "DO NOT COPY", "SAMPLE", "ORIGINAL"],
            )
        else:
            watermark_word = self.watermark_word

        # initialize watermark font size
        watermark_font_size = random.randint(self.watermark_font_size[0], self.watermark_font_size[1])

        # initialize watermark font thickness
        watermark_font_thickness = random.randint(self.watermark_font_thickness[0], self.watermark_font_thickness[1])

        # initialize watermark rotation
        watermark_rotation = random.randint(self.watermark_rotation[0], self.watermark_rotation[1])

        # initialize watermark color
        if self.watermark_color == "random":
            watermark_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            watermark_color = self.watermark_color

        (image_width, image_height), _ = cv2.getTextSize(
            watermark_word,
            self.watermark_font_type,
            watermark_font_size,
            watermark_font_thickness,
        )

        offset = 20 + watermark_font_thickness

        # initialize watermark foreground
        watermark_foreground = np.full((image_height + offset, image_width + offset, 3), fill_value=255, dtype="uint8")

        # draw watermark text
        cv2.putText(
            watermark_foreground,
            watermark_word,
            (int(offset / 2), int(offset / 2) + image_height),
            self.watermark_font_type,
            watermark_font_size,
            watermark_color,
            watermark_font_thickness,
        )

        # rotate image
        watermark_foreground = rotate_image(watermark_foreground, watermark_rotation)

        return watermark_foreground

    # Apply watermark into input image
    def apply_watermark(self, watermark_foreground, image):

        # resize watermark foreground if the size is larger than input image
        ysize, xsize = image.shape[:2]
        ysize_watermark, xsize_watermark = watermark_foreground.shape[:2]
        if ysize_watermark > ysize or xsize_watermark > xsize:
            watermark_foreground = cv2.resize(watermark_foreground, (xsize, ysize), interpolation=cv2.INTER_AREA)

        # initialize watermark location
        if self.watermark_location == "random":
            watermark_location = random.choice(["left", "right", "top", "bottom", "center"])
        else:
            watermark_location = self.watermark_location

        # initialize watermark method
        if self.watermark_method == "random" or self.watermark_method not in ["overlay", "obfuscate"]:
            watermark_method = random.choice(["overlay", "obfuscate"])
        else:
            watermark_method = self.watermark_method

        if watermark_method == "obfuscate":

            # blur image
            image_blurred = cv2.blur(image, (5, 5)).astype("uint8")
            if len(image_blurred.shape) > 2:
                image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)

            # convert to binary
            _, image_binarized = cv2.threshold(
                image_blurred,
                0,
                255,
                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
            )

            # get kernel for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

            # dilate and erode the binary image
            image_dilated = cv2.dilate(
                image_binarized,
                kernel,
                iterations=2,
            )
            image_eroded = cv2.erode(
                image_dilated,
                None,
                iterations=1,
            )

            # get contours
            contours, hierarchy = cv2.findContours(
                image_eroded,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            # complement image
            image_eroded = 255 - image_eroded
            # remove contours in fixed interval
            for i, contour in enumerate(contours):
                if i % 2:
                    x, y, w, h = cv2.boundingRect(contour)
                    image_eroded[y : y + h, x : x + w] = 255

            # create blank image and overlay the foreground only
            image_blank = np.full_like(image, fill_value=255, dtype="uint8")
            ob = OverlayBuilder(
                "darken",
                watermark_foreground,
                image_blank,
                ntimes=1,
                nscales=(1, 1),
                edge=watermark_location,
                edge_offset=10,
            )
            new_watermark_foreground = ob.build_overlay()

            # set removed contours to white
            if len(new_watermark_foreground.shape) > 2:
                image_eroded = cv2.cvtColor(image_eroded, cv2.COLOR_GRAY2BGR)

            new_watermark_foreground[image_eroded == 0] = 255
            image[new_watermark_foreground < 255] = 255

            # overlay watermark foreground and input image
            ob = OverlayBuilder(
                "darken",
                new_watermark_foreground,
                image,
                ntimes=1,
                nscales=(1, 1),
                edge="center",
                edge_offset=0,
                alpha=0.5,
            )

        else:

            # overlay watermark foreground and input image
            ob = OverlayBuilder(
                "darken",
                watermark_foreground,
                image,
                ntimes=1,
                nscales=(1, 1),
                edge=watermark_location,
                edge_offset=10,
                alpha=0.5,
            )

        return ob.build_overlay()

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            watermark_foreground = self.create_watermark(image)
            watermark_image = self.apply_watermark(watermark_foreground, image)

            return watermark_image

import os
import random

import cv2
import numpy as np

from augraphy import *
from augraphy.augmentations.lib import add_noise
from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class BindingsAndFasteners(Augmentation):
    """Creates binding and fastener mark in the input image.

    :param overlay_types: Types of overlay method.
    :type overlay_types: string
    :param foreground: Path to foreground image or the foreground image.
    :type foreground: string or numpy array, optional
    :param effect_type: Types of binding effect, select from either
        "random", "punch_holes", binding_holes" or "clips".
    :type effect_type: string, optional
    :param ntimes: Pair of ints to determine number of repetition to draw foreground image.
    :type ntimes: tuple, optional
    :param nscales: Pair of floats to determine scale of foreground image size.
    :type nscales: tuple, optional
    :param edge: Which edge of the page the foreground copies should be
        placed on.
    :type edge: string, optional
    :param edge_offset: Pair of ints to determine how far from the edge of the page to draw the copies.
    :type edge_offset: tuple, optional
    :param use_figshare_library: Flag to download foreground images from figshare library.
    :type use_figshare_library: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        overlay_types="random",
        foreground=None,
        effect_type="random",
        ntimes=(2, 6),
        nscales=(1.0, 1.5),
        edge="random",
        edge_offset=(5, 20),
        use_figshare_library=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.overlay_types = overlay_types
        self.foreground = foreground
        self.effect_type = effect_type
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edge_offset = edge_offset
        self.use_figshare_library = use_figshare_library

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BindingsAndFasteners(overlay_types={self.overlay_types}, foreground={self.foreground}, effect_type={self.effect_type}, ntimes={self.ntimes}, nscales={self.nscales}, edge={self.edge}, edge_offset={self.edge_offset}, use_figshare_library={self.use_figshare_library}, p={self.p})"

    def create_foreground(self, image):

        ysize, xsize = image.shape[:2]

        if self.effect_type == "random":
            effect_type = random.choice(("punch_holes", "binding_holes", "clips"))
        else:
            effect_type == self.effect_type

        if effect_type == "punch_holes":

            template_size = template_size_ori = 30
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 30
            # use max to prevent small template and min to prevent large template
            template_size = int(max(template_size_ori / 2, 30 * ((ysize * xsize) / (1000 * 800))))
            template_size = int(min(template_size, template_size_ori * 2))

            # draw circle
            image_circle = np.zeros((template_size, template_size), dtype="uint8")
            circle_centroid = (int(template_size / 2), int(template_size / 2))
            circle_radius = int(template_size / 2) - 5
            cv2.circle(image_circle, circle_centroid, circle_radius, 255, -1)

            # applies noise
            image_circle = add_noise(image_circle, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_circle = 255 - image_circle
            image_circle = add_noise(image_circle, intensity_range=(0.3, 0.3), color_range=(0, 255))

            # gaussian blur
            image_circle = cv2.GaussianBlur(image_circle.astype("uint8"), (3, 3), cv2.BORDER_DEFAULT)

            # convert to bgr
            image_circle_bgr = np.zeros((template_size, template_size, 3), dtype="uint8")
            image_circle_bgr[:, :, 0] = image_circle.copy()
            image_circle_bgr[:, :, 1] = image_circle.copy()
            image_circle_bgr[:, :, 2] = image_circle.copy()

            self.foreground = image_circle_bgr

        elif effect_type == "binding_holes":

            template_size = template_size_ori = 40
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 40
            # use max to prevent small template and min to prevent large template
            template_size = int(max(template_size_ori / 2, 40 * ((ysize * xsize) / (1000 * 800))))
            template_size = int(min(template_size, template_size_ori * 2))

            # draw rectangle
            image_rectangle = np.zeros((template_size, int(template_size / 2)), dtype="uint8")
            image_rectangle[3:-3:, 4:-4] = 255

            # applies noise
            image_rectangle = add_noise(image_rectangle, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_rectangle = 255 - image_rectangle
            image_rectangle = add_noise(image_rectangle, intensity_range=(0.3, 0.3), color_range=(0, 255))

            # gaussian blur
            image_rectangle = cv2.GaussianBlur(image_rectangle.astype("uint8"), (3, 3), cv2.BORDER_DEFAULT)

            # convert to bgr
            image_rectangle_bgr = np.zeros((template_size, int(template_size / 2), 3), dtype="uint8")
            image_rectangle_bgr[:, :, 0] = image_rectangle.copy()
            image_rectangle_bgr[:, :, 1] = image_rectangle.copy()
            image_rectangle_bgr[:, :, 2] = image_rectangle.copy()

            self.foreground = image_rectangle_bgr

        elif effect_type == "clips":

            # minimum size
            template_size = template_size_ori = 60
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 60
            # use max to prevent small template and min to prevent large template
            template_size = int(max(template_size_ori / 2, 60 * ((ysize * xsize) / (1000 * 800))))
            template_size = int(min(template_size, template_size_ori * 2))

            template_size_y = int(template_size / 3)
            template_size_x = template_size

            # draw line, triangle & circle to create clip effect
            image_clip = np.zeros((template_size_y, template_size_x), dtype="uint8")

            # draw line
            image_clip[3:5:, 5:-3] = 255

            # draw triangle
            pt1, pt2, pt3 = (5, 3), (template_size_x - 4, 3 + int(template_size_y / 2)), (5, template_size_y - 3)
            triangle_contour = np.array([pt1, pt2, pt3])
            cv2.drawContours(image_clip, [triangle_contour], 0, 255, -1)

            # draw circle
            circle_radius = int((pt3[1] - pt1[1]) / 2)
            circle_centroid = (8, 3 + int(template_size_y / 2) - 3)
            cv2.circle(image_clip, circle_centroid, circle_radius, 255, -1)

            # applies noise
            image_clip = add_noise(image_clip, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_clip = 255 - image_clip
            image_clip = add_noise(image_clip, intensity_range=(0.3, 0.3), color_range=(0, 255))

            # gaussian blur
            image_clip = cv2.GaussianBlur(image_clip.astype("uint8"), (3, 3), cv2.BORDER_DEFAULT)

            # convert to bgr
            image_clip_bgr = np.zeros((template_size_y, template_size_x, 3), dtype="uint8")
            image_clip_bgr[:, :, 0] = image_clip.copy()
            image_clip_bgr[:, :, 1] = image_clip.copy()
            image_clip_bgr[:, :, 2] = image_clip.copy()

            self.foreground = image_clip_bgr

    def retrieve_foreground(self):

        # Id for figshare published template files
        article_ID = "16668964"
        # create figshare downloader
        fsdl = FigshareDownloader(directory="figshare_BindingsAndFasteners/")
        # download files
        fsdl.download_all_files_from_article(article_ID)

        if self.effect_type == "random":
            effect_type = random.choice(("punch_holes", "binding_holes", "clips"))
        else:
            effect_type == self.effect_type

        # read foreground
        if self.effect_type == "punch_holes":
            foreground_path = os.path.join(os.getcwd() + "/figshare_BindingsAndFasteners/punch_hole.png")
        elif self.effect_type == "binding_holes":
            foreground_path = os.path.join(os.getcwd() + "/figshare_BindingsAndFasteners/binding_hole.png")
        elif self.effect_type == "clips":
            foreground_path = os.path.join(os.getcwd() + "/figshare_BindingsAndFasteners/clip.png")
        self.foreground = cv2.imread(foreground_path)

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            # reset foreground when the same class instance called twice
            if not isinstance(self.foreground, str) and not isinstance(self.foreground, np.ndarray):
                self.foreground = None

            image = image.copy()

            # generate randomized overlay types
            if self.overlay_types == "random":
                overlay_types = random.choice(
                    (
                        "min",
                        "max",
                        "mix",
                        "normal",
                        "lighten",
                        "darken",
                        "addition",
                        "screen",
                        "dodge",
                        "multiply",
                        "divide",
                        "hard_light",
                        "grain_merge",
                        "overlay",
                    ),
                )
            else:
                overlay_types = self.overlay_types

            # generate randomized edge
            if self.edge == "random":
                edge = random.choice(("left", "right", "top", "bottom"))
            else:
                edge = self.edge

            # generate randomized ntimes
            ntimes = random.randint(self.ntimes[0], self.ntimes[1])

            # generate randomized offset
            edge_offset = random.randint(self.edge_offset[0], self.edge_offset[1])

            # if user input image path
            if isinstance(self.foreground, str) and os.path.isfile(self.foreground):
                self.foreground = cv2.imread(self.foreground)
                ob = OverlayBuilder(
                    overlay_types,
                    self.foreground,
                    image,
                    ntimes,
                    self.nscales,
                    edge,
                    edge_offset,
                    1,
                )
            # if user input image
            elif isinstance(self.foreground, np.ndarray):
                ob = OverlayBuilder(
                    overlay_types,
                    self.foreground,
                    image,
                    ntimes,
                    self.nscales,
                    edge,
                    edge_offset,
                    1,
                )

            else:
                # user didn't input foreground or not readable file, try to download from Figshare
                use_figshare_library = self.use_figshare_library

                if use_figshare_library:
                    try:
                        self.retrieve_foreground()
                        ob = OverlayBuilder(
                            overlay_types,
                            self.foreground,
                            image,
                            ntimes,
                            self.nscales,
                            edge,
                            edge_offset,
                            1,
                        )
                    # if failed to download from Figshare, set to create own foreground
                    except Exception:
                        use_figshare_library = 0

                if not use_figshare_library:
                    self.create_foreground(image)
                    ob = OverlayBuilder(
                        overlay_types,
                        self.foreground,
                        image,
                        ntimes,
                        self.nscales,
                        edge,
                        edge_offset,
                        1,
                    )

            image_output = ob.build_overlay()

            return image_output

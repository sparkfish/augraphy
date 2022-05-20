import os
import random

import cv2
import numpy as np

from augraphy import *
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

    # Add noise to image
    def add_noise(self, image, noise_probability, noise_value, max_input_value):

        noise = (
            lambda x: random.randint(noise_value[0], noise_value[1])
            if (x < max_input_value and noise_probability > random.random())
            else x
        )
        add_noise = np.vectorize(noise)
        image_output = add_noise(image)
        return image_output

    def create_foreground(self, image):

        ysize, xsize = image.shape[:2]

        if self.effect_type == "random":
            effect_type = random.choice(("punch_holes", "binding_holes", "clips"))
        else:
            effect_type = self.effect_type

        if effect_type == "punch_holes":

            self.foreground = []
            template_size = template_size_ori = 60
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 30
            # use max to prevent small template and min to prevent large template
            template_size = int(
                max(template_size_ori / 4, 30 * ((ysize * xsize) / (900 * 700))),
            )
            template_size = int(min(template_size, template_size_ori * 2))

            # create random location to merge 2 circles
            min_value = min(10, int(template_size / 2))
            random_x = random.randint(min_value, template_size - min_value)
            random_y = random.randint(min_value, template_size - min_value)

            # number of random rotation
            rotate_num = random.randint(1, 3)

            for _ in range(self.ntimes[1]):
                # draw circle
                image_circle = np.full(
                    (template_size, template_size),
                    fill_value=255,
                    dtype="uint8",
                )
                circle_centroid = (int(template_size / 2), int(template_size / 2))
                circle_radius = max(int(template_size / 4) - 5, 5)
                cv2.circle(image_circle, circle_centroid, circle_radius, 0, -1)

                # add small blob noise effect
                if random.random() > 0.7:
                    angle = random.randint(0, 360)
                    circle_centroid_small = (
                        int(
                            circle_centroid[0] + circle_radius * np.cos(np.radians(angle)),
                        ),
                        int(
                            circle_centroid[1] + circle_radius * np.sin(np.radians(angle)),
                        ),
                    )
                    circle_radius_small = max(
                        int(circle_radius * random.uniform(0.1, 0.5)),
                        2,
                    )
                    cv2.circle(
                        image_circle,
                        circle_centroid_small,
                        circle_radius_small,
                        0,
                        -1,
                    )

                # add noise
                image_circle = self.add_noise(
                    image_circle,
                    random.uniform(0.01, 0.21),
                    (0, 255),
                    10,
                )

                # create another copy of complement image
                image_circle_complement = 255 - image_circle.copy()

                # merge 2 circles to create non-perfect circle effect
                image_circle[random_y:, random_x:] = np.maximum(
                    image_circle[random_y:, random_x:],
                    image_circle_complement[:-random_y, :-random_x],
                )

                # randomly rotate to get different direction effect
                image_circle = np.rot90(image_circle, rotate_num)

                # convert to bgr
                image_circle_bgr = cv2.cvtColor(image_circle, cv2.COLOR_GRAY2BGR)

                self.foreground.append(image_circle_bgr)

        elif effect_type == "binding_holes":

            template_size = template_size_ori = 40
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 40
            # use max to prevent small template and min to prevent large template
            template_size = int(
                max(template_size_ori / 2, 40 * ((ysize * xsize) / (1000 * 800))),
            )
            template_size = int(min(template_size, template_size_ori * 2))

            self.foreground = []

            random_x_offset = random.randint(4, 16)
            random_y_offset = random.randint(2, 6)

            offset_p = random.random() > 0.2
            y_offset_p = random.random() > 0.5

            x_offset_type_p = random.random() > 0.5
            y_offset_type_p = random.random() > 0.5

            random_scale = random.uniform(1, 2)

            for _ in range(self.ntimes[1]):

                # draw rectangle
                offset = int(template_size / random.uniform(7, 8))
                image_rectangle = np.full(
                    (template_size, int(template_size / random_scale)),
                    fill_value=255,
                    dtype="uint8",
                )
                image_rectangle[offset:-offset:, offset:-offset] = 0

                new_offset = offset + random.randint(3, 6)
                image_rectangle_complement = np.full(
                    (template_size, int(template_size / random_scale)),
                    fill_value=0,
                    dtype="uint8",
                )
                image_rectangle_complement[new_offset:-new_offset:, new_offset:-new_offset] = 255

                image_rectangle = self.add_noise(
                    image_rectangle,
                    random.uniform(0.01, 0.21),
                    (0, 255),
                    10,
                )

                # create offset effect
                if offset_p:
                    # x offset is default
                    if x_offset_type_p:
                        image_rectangle_complement[:, :-random_x_offset] = image_rectangle_complement[
                            :,
                            random_x_offset:,
                        ]
                    else:
                        image_rectangle_complement[:, random_x_offset:] = image_rectangle_complement[
                            :,
                            :-random_x_offset,
                        ]

                    # create y offset effect
                    if y_offset_p:
                        if y_offset_type_p:
                            image_rectangle_complement[:-random_y_offset, :] = image_rectangle_complement[
                                random_y_offset:,
                                :,
                            ]
                        else:
                            image_rectangle_complement[random_y_offset:, :] = image_rectangle_complement[
                                :-random_y_offset,
                                :,
                            ]
                    # merge 2 image to create offset effect
                    image_rectangle = np.maximum(
                        image_rectangle,
                        image_rectangle_complement,
                    )

                # add noise and apply blur
                image_rectangle = self.add_noise(
                    image_rectangle,
                    random.uniform(0.01, 0.21),
                    (0, 255),
                    10,
                )
                image_rectangle = cv2.GaussianBlur(
                    image_rectangle.astype("uint8"),
                    (3, 3),
                    cv2.BORDER_DEFAULT,
                )

                # convert to bgr
                image_rectangle_bgr = cv2.cvtColor(image_rectangle, cv2.COLOR_GRAY2BGR)

                self.foreground.append(image_rectangle_bgr)

        elif effect_type == "clips":

            # minimum size
            template_size = template_size_ori = 60
            # scale template size based on image size
            # 1000 * 800 is normal image size for template size = 60
            # use max to prevent small template and min to prevent large template
            template_size = int(
                max(template_size_ori / 2, 60 * ((ysize * xsize) / (1000 * 800))),
            )
            template_size = int(min(template_size, template_size_ori * 2))

            template_size_y = int(template_size / 3)
            template_size_x = template_size

            self.foreground = []

            for _ in range(self.ntimes[1]):

                # draw line, triangle & circle to create clip effect
                image_clip = np.full(
                    (template_size_y, template_size_x),
                    fill_value=255,
                    dtype="uint8",
                )

                # draw triangle
                pt1, pt2, pt3 = (
                    (5, 3),
                    (template_size_x - 4, 3 + int(template_size_y / 2)),
                    (5, template_size_y - 3),
                )
                triangle_contour = np.array([pt1, pt2, pt3])
                cv2.drawContours(image_clip, [triangle_contour], 0, 0, -1)

                # draw circle
                circle_radius = int((pt3[1] - pt1[1]) / 2)
                circle_centroid = (8, 3 + int(template_size_y / 2) - 3)
                cv2.circle(image_clip, circle_centroid, circle_radius, 0, -1)

                # add noise
                image_clip = self.add_noise(
                    image_clip,
                    random.uniform(0.01, 0.21),
                    (0, 255),
                    10,
                )
                image_clip_complement = 255 - image_clip

                # create cip inner part
                scale_percent = random.uniform(0.5, 0.8)
                width = int(image_clip_complement.shape[1] * scale_percent)
                height = int(image_clip_complement.shape[0] * scale_percent)
                dimension = (width, height)

                # resize image
                image_clip_complement_resize = cv2.resize(
                    image_clip_complement,
                    dimension,
                    interpolation=cv2.INTER_AREA,
                )
                ysize, xsize = image_clip_complement_resize.shape[:2]

                # get offset
                offset = int((template_size - (template_size * scale_percent)) / 4)
                image_crop = image_clip_complement[offset : offset + ysize, offset : offset + xsize]

                # draw clip inner part
                if image_crop.shape[0] == ysize and image_crop.shape[1] == xsize:
                    image_clip_complement[:] = 0
                    image_clip_complement[
                        offset : offset + ysize,
                        offset : offset + xsize,
                    ] = image_clip_complement_resize

                # draw line
                image_clip[3:5:, random.randint(2, 6) : -(random.randint(2, 6))] = 0

                #  add noise and apply blur
                image_clip = self.add_noise(
                    image_clip,
                    random.uniform(0.01, 0.21),
                    (0, 255),
                    10,
                )
                image_clip = cv2.GaussianBlur(
                    image_clip.astype("uint8"),
                    (3, 3),
                    cv2.BORDER_DEFAULT,
                )

                # merge image
                image_clip = np.maximum(image_clip, image_clip_complement)

                # convert to bgr
                image_clip_bgr = cv2.cvtColor(image_clip, cv2.COLOR_GRAY2BGR)

                if random.random() > 0.5:
                    image_clip_bgr = np.fliplr(image_clip_bgr)

                self.foreground.append(image_clip_bgr)

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
            effect_type = self.effect_type

        # read foreground
        if self.effect_type == "punch_holes":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/punch_hole.png",
            )
        elif self.effect_type == "binding_holes":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/binding_hole.png",
            )
        elif self.effect_type == "clips":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/clip.png",
            )
        self.foreground = cv2.imread(foreground_path)

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():

            # reset foreground when the same class instance called twice
            if not isinstance(self.foreground, str) and not isinstance(
                self.foreground,
                np.ndarray,
            ):
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

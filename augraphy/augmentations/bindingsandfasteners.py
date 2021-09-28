import os

import cv2
import numpy as np

from augraphy import *
from augraphy.augmentations.lib import addNoise
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.utilities import *


class BindingsAndFasteners(Augmentation):
    """Creates binding and fastener mark in the input image.
    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param foreground: Path to foreground image.
    :type foreground: string, optional
    :param effect_type: Types of binding effect, select from either
        "punch_holes", binding_holes" or "clips".
    :type effect_type: string, optional
    :param ntimes: Number of repetition to draw foreground image.
    :type ntimes: int, optional
    :param edge: Which edge of the page the foreground copies should be
        placed on.
    :type edge: string, optional
    :param edgeOffset: How far from the edge of the page to draw the copies.
    :type edgeOffset: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        foreground=None,
        effect_type="punch_holes",
        ntimes=3,
        edge="left",
        edgeOffset=50,
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.layer = layer
        self.foreground = foreground
        self.effect_type = effect_type
        self.ntimes = ntimes
        self.edge = edge
        self.edgeOffset = max(0, edgeOffset)  # prevent negative

        # check for valid effect types
        if self.effect_type not in ["punch_holes", "binding_holes", "clips"]:
            self.effect_type = "punch_holes"
            # print warning here on the invalid effect type

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BindingsAndFasteners({self.layer}, {self.foreground}, effect_type={self.effect_type}, ntimes={self.ntimes}, edge={self.edge}, edgeOffset={self.edgeOffset}, p={self.p})"

    def create_foreground(self, image):

        ysize, xsize = image.shape[:2]

        if self.effect_type == "punch_holes":

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
            image_circle = addNoise(image_circle, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_circle = 255 - image_circle
            image_circle = addNoise(image_circle, intensity_range=(0.3, 0.3), color_range=(0, 255))

            # gaussian blur
            image_circle = cv2.GaussianBlur(image_circle.astype("uint8"), (3, 3), cv2.BORDER_DEFAULT)

            # convert to bgr
            image_circle_bgr = np.zeros((template_size, template_size, 3), dtype="uint8")
            image_circle_bgr[:, :, 0] = image_circle.copy()
            image_circle_bgr[:, :, 1] = image_circle.copy()
            image_circle_bgr[:, :, 2] = image_circle.copy()

            self.foreground = image_circle_bgr

        elif self.effect_type == "binding_holes":

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
            image_rectangle = addNoise(image_rectangle, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_rectangle = 255 - image_rectangle
            image_rectangle = addNoise(image_rectangle, intensity_range=(0.3, 0.3), color_range=(0, 255))

            # gaussian blur
            image_rectangle = cv2.GaussianBlur(image_rectangle.astype("uint8"), (3, 3), cv2.BORDER_DEFAULT)

            # convert to bgr
            image_rectangle_bgr = np.zeros((template_size, int(template_size / 2), 3), dtype="uint8")
            image_rectangle_bgr[:, :, 0] = image_rectangle.copy()
            image_rectangle_bgr[:, :, 1] = image_rectangle.copy()
            image_rectangle_bgr[:, :, 2] = image_rectangle.copy()

            self.foreground = image_rectangle_bgr

        elif self.effect_type == "clips":

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
            image_clip = addNoise(image_clip, intensity_range=(0.05, 0.05), color_range=(0, 255))
            image_clip = 255 - image_clip
            image_clip = addNoise(image_clip, intensity_range=(0.3, 0.3), color_range=(0, 255))

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
        fsdl = FigshareDownloader()
        # download files
        fsdl.downloadAllFiles(article_ID)

        # read foreground
        if self.effect_type == "punch_holes":
            foreground_path = os.path.join(os.getcwd() + "/figshare/punch_hole.png")
        elif self.effect_type == "binding_holes":
            foreground_path = os.path.join(os.getcwd() + "/figshare/binding_hole.png")
        elif self.effect_type == "clips":
            foreground_path = os.path.join(os.getcwd() + "/figshare/clip.png")
        self.foreground = cv2.imread(foreground_path)

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result

            # if user input image
            if self.foreground and os.path.isfile(self.foreground):
                self.foreground = cv2.imread(self.foreground)
                ob = OverlayBuilder(self.foreground, image, self.ntimes, self.edge, self.edgeOffset)
            else:
                # user didn't input foreground or not readable file, try to download from Figshare
                try:
                    self.retrieve_foreground()
                    ob = OverlayBuilder(self.foreground, image, self.ntimes, self.edge, self.edgeOffset)
                # if failed to download from Figshare, create some simple effect
                except Exception:
                    self.create_foreground(image)
                    ob = OverlayBuilder(self.foreground, image, self.ntimes, self.edge, self.edgeOffset)

            image_output = ob.buildOverlay()

            data[self.layer].append(AugmentationResult(self, image_output))

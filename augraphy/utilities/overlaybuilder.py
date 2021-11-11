import math
import random

import cv2
import numpy as np


class OverlayBuilder:
    """Takes an input image, a number of times to duplicate that image, an image
    on which to overlay the result of this, and a page position, then produces
    an overlayable image with the input image copied that many times across
    the edge of the page or at random location or at the center of image.

    :param overlay_types: Types of overlay method, min, max or mix.
    :type overlay_types: string
    :param foreground: the image to overlay on the background document
    :type foreground: np.array
    :param background: the document
    :type background: np.array
    :param ntimes: how many copies of the foreground image to draw
    :type ntimes: integer
    :param nscales: Scales of foreground image size.
    :type nscales: tuple, optional
    :param edge: which edge of the page the foreground copies should be
        placed on. Selections included left, right, top, bottom, enter, random.
    :type edge: string
    :param edge_offset: how far from the edge of the page to draw the copies
    :type edge_offset: integer
    """

    def __init__(self, overlay_types, foreground, background, ntimes, nscales, edge, edge_offset):
        self.overlay_types = overlay_types
        self.foreground = foreground
        self.background = background
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edge_offset = max(0, edge_offset)  # prevent negative

        # set valid overlay types
        if overlay_types != "mix" and overlay_types != "min" and overlay_types != "min":
            self.overlay_types = "mix"

    def compute_offsets(self):
        """Determine where to place the foreground image copies"""
        xdim = self.background.shape[1]
        ydim = self.background.shape[0]

        img_width = self.foreground.shape[1]
        img_height = self.foreground.shape[0]

        remaining_width = xdim - (self.ntimes * img_width)
        remaining_height = ydim - (self.ntimes * img_height)

        # max to prevent negative offset
        offset_width = max(0, math.floor(remaining_width / (self.ntimes + 1)))
        offset_height = max(0, math.floor(remaining_height / (self.ntimes + 1)))

        return offset_width, offset_height

    def check_size(self, img_foreground, img_background, center=None):
        """Check the fitting size of foreground to background"""

        # background size
        ysize_background, xsize_background = img_background.shape[:2]

        # get center x and y
        center_x, center_y = center

        # foreground size
        ysize_foreground, xsize_foreground = img_foreground.shape[:2]

        # center point of foreground
        ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)

        # if foreground size is > background size, crop only the fitting size
        if center_y - ysize_half_foreground < 0 and center_y + ysize_half_foreground > ysize_background:
            img_foreground = img_foreground[
                -(center_y - ysize_half_foreground) : ysize_foreground
                - (center_y + ysize_half_foreground - ysize_background),
                :,
            ]
            # new size after cropping
            # foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # center point of foreground
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)

        if center_x - xsize_half_foreground < 0 and center_x + xsize_half_foreground > xsize_background:
            img_foreground = img_foreground[
                :,
                -(center_x - xsize_half_foreground) : xsize_foreground
                - (center_x + xsize_half_foreground - xsize_background),
            ]
            # new size after cropping
            # foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # center point of foreground
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)

        # to prevent having no overlap between foreground and background image
        # check width max size
        if center_x - xsize_half_foreground >= xsize_background:
            # at least 10 pixel overlapping area
            center_x = xsize_background + xsize_half_foreground - 10
        # check width min size
        elif center_x + xsize_half_foreground < 0:
            # at least 10 pixel overlapping area
            center_x = 10 - xsize_half_foreground
        # check height max size
        if center_y - ysize_half_foreground >= ysize_background:
            # at least 10 pixel overlapping area
            center_y = ysize_background + ysize_half_foreground - 10
        # check height min size
        elif center_y + ysize_half_foreground < 0:
            # at least 10 pixel overlapping area
            center_y = 10 - ysize_half_foreground

        # if foreground x exceed background width
        if center_x + xsize_half_foreground > xsize_background:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[:, : -(center_x + xsize_half_foreground - xsize_background)]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)
            # update new center
            center = [xsize_background - xsize_half_foreground, center[1]]

        # if foreground x < 0
        if center_x - xsize_half_foreground < 0:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[:, abs(center_x - xsize_half_foreground) :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)
            # update new center
            center = [xsize_half_foreground, center[1]]

        # if foreground y exceed background height
        if center_y + ysize_half_foreground > ysize_background:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[: -(center_y + ysize_half_foreground - ysize_background), :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)

            # update new center
            center = [center[0], ysize_background - ysize_half_foreground]

        # if foreground y < 0
        if center_y - ysize_half_foreground < 0:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[abs(center_y - ysize_half_foreground) :, :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(xsize_foreground / 2)
            # update new center
            center = [center[0], ysize_half_foreground]

        return img_foreground, center

    def apply_overlay(
        self,
        overlay_background,
        offset_width,
        offset_height,
        ystart,
        yend,
        xstart,
        xend,
    ):
        """Applies overlay from foreground to background"""

        # convert background to gray
        if len(overlay_background.shape) > 2:
            overlay_background_gray = cv2.cvtColor(overlay_background, cv2.COLOR_BGR2GRAY)
        else:
            overlay_background_gray = overlay_background

        # convert foreground to gray
        if len(self.foreground.shape) > 2:
            foreground_gray = cv2.cvtColor(self.foreground, cv2.COLOR_BGR2GRAY)
        else:
            foreground_gray = self.foreground

        # get size
        bg_height, bg_width = overlay_background_gray.shape
        fg_height, fg_width = foreground_gray.shape

        for i in range(self.ntimes):

            if self.edge == "random":
                ystart = random.randint(0, bg_height - 10)
                yend = ystart + fg_height
                xstart = random.randint(0, bg_width - 10)
                xend = xstart + fg_width

            # crop a section of background
            base = overlay_background[ystart:yend, xstart:xend]
            base_gray = overlay_background_gray[ystart:yend, xstart:xend]

            # center of overlay
            center = (xstart + int(fg_width / 2), ystart + int(fg_height / 2))

            # check for size mismatch issue
            new_foreground, center = self.check_size(self.foreground, overlay_background, center)

            # convert foreground to gray
            if len(new_foreground.shape) > 2:
                new_foreground_gray = cv2.cvtColor(new_foreground, cv2.COLOR_BGR2GRAY)
            else:
                new_foreground_gray = new_foreground

            # new foreground height and width
            fg_height, fg_width = new_foreground_gray.shape

            if self.overlay_types == "min" or self.overlay_types == "max":
                # can be further vectorized here, need to think about it
                for y in range(fg_height):
                    for x in range(fg_width):
                        if self.overlay_types == "min":
                            check_condition = new_foreground_gray[y, x] < base_gray[y, x]
                        else:
                            check_condition = new_foreground_gray[y, x] > base_gray[y, x]

                        if check_condition:  # foreground is darker, get value from foreground
                            # foreground is colour but base in gray
                            if len(new_foreground.shape) > len(base.shape):
                                base[y, x] = new_foreground_gray[y, x]
                            # same channel number
                            else:
                                base[y, x] = new_foreground[y, x]

            elif self.overlay_types == "mix":

                img_mask = np.ones((fg_height, fg_width), dtype="uint8") * 255
                overlay_background = cv2.seamlessClone(
                    new_foreground,
                    overlay_background,
                    img_mask,
                    center,
                    cv2.MIXED_CLONE,
                )

            # get original height and width from foreground
            fg_height, fg_width = self.foreground.shape[:2]

            if self.edge == "left" or self.edge == "right":
                # for next loop ystart and yend
                ystart += fg_height + offset_height
                yend = ystart + fg_height
                if self.overlay_types == "min" or self.overlay_types == "max":
                    # break when next ystart is > image y size
                    if ystart >= bg_height - fg_height:
                        break

            elif self.edge == "top" or self.edge == "bottom":
                # for next loop xstart and xend
                xstart += fg_width + offset_width
                xend = xstart + fg_width
                if self.overlay_types == "min" or self.overlay_types == "max":
                    # break when next xstart is > image x size
                    if xstart >= bg_width - fg_width:
                        break

        return overlay_background

    def build_overlay(self):
        """Construct the overlay image containing foreground copies"""

        overlay_background = self.background

        # sensitivity up to 0.01 (for scale down)
        min_value = max(1, self.nscales[0] * 100)
        max_value = max(1, self.nscales[1] * 100)
        new_fg_height = int((self.foreground.shape[0] * random.randint(min_value, max_value)) / 100)
        new_fg_width = int((self.foreground.shape[1] * random.randint(min_value, max_value)) / 100)
        self.foreground = cv2.resize(
            self.foreground,
            (int(new_fg_width), int(new_fg_height)),
            interpolation=cv2.INTER_AREA,
        )

        # foreground size (height & width)
        fg_height, fg_width = self.foreground.shape[:2]

        # background size (height & width)
        bg_height, bg_width = self.background.shape[:2]

        # compute offsets between foreground and background
        offset_width, offset_height = self.compute_offsets()

        # get overlay location for each types of edge
        if self.edge == "left":
            ystart = offset_height
            yend = ystart + fg_height
            xstart = self.edge_offset
            xend = self.edge_offset + fg_width

        elif self.edge == "right":
            ystart = offset_height
            yend = ystart + fg_height
            xstart = bg_width - self.edge_offset - fg_width
            xend = bg_width - self.edge_offset

        elif self.edge == "top":
            ystart = self.edge_offset
            yend = self.edge_offset + fg_height
            xstart = offset_width
            xend = offset_width + fg_width

        elif self.edge == "bottom":
            ystart = bg_height - self.edge_offset - fg_height
            yend = bg_height - self.edge_offset
            xstart = offset_width
            xend = offset_width + fg_width

        elif self.edge == "random":
            ystart = random.randint(0, bg_height - 10)
            yend = ystart + fg_height
            xstart = random.randint(0, bg_width - 10)
            xend = xstart + fg_width

        elif self.edge == "center":
            ystart = int(bg_height / 2) - int(fg_height / 2)
            yend = ystart + fg_height
            xstart = int(bg_width / 2) - int(fg_width / 2)
            xend = xstart + fg_width

        # apply overlay
        overlay_background = self.apply_overlay(
            overlay_background,
            offset_width,
            offset_height,
            ystart,
            yend,
            xstart,
            xend,
        )

        return overlay_background


# usage example
if __name__ == "__main__":

    import os
    from matplotlib import pyplot as plt
    from augraphy.utilities import FigshareDownloader

    # Id for figshare published template files
    article_ID = "16668964"
    # create figshare downloader
    fsdl = FigshareDownloader()
    # download files
    fsdl.downloadAllFiles(article_ID)

    # donwload foreground
    foreground_path = os.path.join(os.getcwd() + "/figshare/clip.png")
    img_foreground = cv2.imread(foreground_path)

    # create background (black stripes image)
    img_background = np.ones((1000, 1000, 3), dtype="uint8") * 255
    img_background[4::4, :] = 0

    # create overlay object
    ob_mix = OverlayBuilder("mix", img_foreground, img_background.copy(), 4, (3, 5), "random", 0)
    ob_min = OverlayBuilder("min", img_foreground, img_background.copy(), 4, (3, 5), "center", 0)
    ob_max = OverlayBuilder("max", img_foreground, img_background.copy(), 4, (3, 5), "left", 10)

    # performs overlay
    image_output_mix = ob_mix.build_overlay()
    image_output_min = ob_min.build_overlay()
    image_output_max = ob_max.build_overlay()

    # plot examples
    plt.figure()
    plt.imshow(cv2.cvtColor(image_output_mix, cv2.COLOR_BGR2RGB))
    plt.title("Mix blend and random location")

    plt.figure()
    plt.imshow(cv2.cvtColor(image_output_min, cv2.COLOR_BGR2RGB))
    plt.title("Min blend and blend at center")

    plt.figure()
    plt.imshow(cv2.cvtColor(image_output_max, cv2.COLOR_BGR2RGB))
    plt.title("Max blend and blend at left side")

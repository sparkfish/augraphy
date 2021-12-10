import math
import random

import cv2
import numpy as np


class OverlayBuilder:
    """Takes an input image, a number of times to duplicate that image, an image
    on which to overlay the result of this, and a page position, then produces
    an overlayable image with the input image copied that many times across
    the edge of the page or at random location or at the center of image.

    :param overlay_types: Types of overlay method.
    :type overlay_types: string
    :param foreground: The image to overlay on the background document.
    :type foreground: np.array
    :param background: The document.
    :type background: np.array
    :param ntimes: Number copies of the foreground image to draw.
    :type ntimes: integer
    :param nscales: Scales of foreground image size.
    :type nscales: tuple, optional
    :param edge: Which edge of the page the foreground copies should be
        placed on. Selections included left, right, top, bottom, enter, random.
    :type edge: string
    :param edge_offset: How far from the edge of the page to draw the copies.
    :type edge_offset: integer
    :param alpha: Alpha value for alpha overlay type.
    :type alpha: float
    """

    def __init__(
        self,
        overlay_types,
        foreground,
        background,
        ntimes=1,
        nscales=(1, 1),
        edge="center",
        edge_offset=0,
        alpha=0.3,
    ):
        self.overlay_types = overlay_types
        self.foreground = foreground
        self.background = background
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edge_offset = max(0, edge_offset)  # prevent negative
        self.alpha = alpha

        # set valid edge type
        if edge not in ["center", "random", "left", "right", "top", "bottom"]:
            self.edge = "center"

        # set valid overlay types
        if overlay_types not in [
            "min",
            "max",
            "mix",
            "normal",
            "lighten",
            "darken",
            "addition",
            "subtract",
            "difference",
            "screen",
            "dodge",
            "multiply",
            "divide",
            "hard_light",
            "grain_extract",
            "grain_merge",
            "overlay",
        ]:
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

    def compose_alpha(self, img_alpha_background, img_alpha_foreground):
        """Calculate alpha composition ratio between two images."""

        comp_alpha = np.minimum(img_alpha_background, img_alpha_foreground) * self.alpha
        new_alpha = img_alpha_background + (1.0 - img_alpha_foreground) * comp_alpha
        np.seterr(divide="ignore", invalid="ignore")
        ratio = comp_alpha / new_alpha
        ratio[ratio == np.NAN] = 0.0
        return ratio

    def mix_blend(self, overlay_background, new_foreground, center, fg_height, fg_width):
        """Apply blending using cv2.seamlessClone"""

        img_mask = np.ones((fg_height, fg_width), dtype="uint8") * 255

        # convert gray to bgr (seamlessClone need bgr format)
        if len(new_foreground.shape) < 3:
            new_foreground = cv2.cvtColor(new_foreground, cv2.COLOR_GRAY2BGR)
        if len(overlay_background.shape) < 3:
            overlay_background = cv2.cvtColor(overlay_background, cv2.COLOR_GRAY2BGR)

        overlay_background = cv2.seamlessClone(
            new_foreground,
            overlay_background,
            img_mask,
            center,
            cv2.MIXED_CLONE,
        )

        # convert from bgr back to gray
        if len(new_foreground.shape) < 3:
            overlay_background = cv2.cvtColor(new_foreground, cv2.COLOR_BGR2GRAY)

        return overlay_background

    def min_max_blend(self, base, base_gray, new_foreground, new_foreground_gray, fg_height, fg_width):
        """Apply blending using min or max gray value"""

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

    def normal_blend(self, overlay_background, base, new_foreground, xstart, xend, ystart, yend):
        """Apply blending using input alpha value - normal method"""

        # convert to float (0-1)
        base_norm = base / 255.0
        foreground_norm = new_foreground / 255.0

        # add alpha value of base and foreground
        img_base_alpha = np.zeros_like(base_norm) + 1.0
        img_foreground_alpha = np.zeros_like(foreground_norm) + self.alpha

        # blend base and foreground
        img_blended = (foreground_norm * img_foreground_alpha) + (
            base_norm * img_base_alpha * (1 - img_foreground_alpha)
        )

        # normalized by alpha value
        img_blended_norm = img_blended / (img_foreground_alpha + img_base_alpha * (1 - img_foreground_alpha))

        # convert blended image back to uint8
        img_blended_norm = (img_blended_norm * 255.0).astype("uint8")

        # add patch of blended image back to background
        overlay_background[ystart:yend, xstart:xend] = img_blended_norm

    def various_blend(self, overlay_background, base, new_foreground, xstart, xend, ystart, yend):
        """Apply blending using input alpha value - multiple methods"""

        # convert to float (0-1)
        base_norm = base / 255.0
        foreground_norm = new_foreground / 255.0

        # get alpha layer (if any)
        if len(base_norm.shape) > 3:
            img_base_alpha = base_norm[:, :, 3]
        else:
            img_base_alpha = np.ones((base_norm.shape[0], base_norm.shape[1]), dtype="float")

        # get alpha layer (if any)
        if len(foreground_norm.shape) > 3:
            img_foreground_alpha = foreground_norm[:, :, 3]
        else:
            img_foreground_alpha = np.ones((foreground_norm.shape[0], foreground_norm.shape[1]), dtype="float")

        # compose alpha ratio from background and foreground alpha value
        ratio = self.compose_alpha(img_base_alpha, img_foreground_alpha)

        # compute alpha value
        if self.overlay_types == "lighten":
            comp_value = np.maximum(base_norm[:, :, :3], foreground_norm[:, :, :3])

        elif self.overlay_types == "darken":
            comp_value = np.minimum(base_norm[:, :, :3], foreground_norm[:, :, :3])

        elif self.overlay_types == "addition":
            comp_value = base_norm[:, :, :3] + foreground_norm[:, :, :3]

        elif self.overlay_types == "subtract":
            comp_value = base_norm[:, :, :3] - foreground_norm[:, :, :3]

        elif self.overlay_types == "difference":
            comp_value = abs(base_norm[:, :, :3] - foreground_norm[:, :, :3])

        elif self.overlay_types == "screen":
            comp_value = 1.0 - (1.0 - base_norm[:, :, :3]) * (1.0 - foreground_norm[:, :, :3])

        elif self.overlay_types == "dodge":
            comp_value = np.minimum(base_norm[:, :, :3] / (1.0 - foreground_norm[:, :, :3]), 1.0)

        elif self.overlay_types == "multiply":
            comp_value = np.clip(base_norm[:, :, :3] * foreground_norm[:, :, :3], 0.0, 1.0)

        elif self.overlay_types == "divide":
            comp_value = np.minimum(
                (256.0 / 255.0 * base_norm[:, :, :3]) / (1.0 / 255.0 + foreground_norm[:, :, :3]),
                1.0,
            )

        elif self.overlay_types == "hard_light":
            base_greater = np.greater(base_norm[:, :, :3], 0.5)
            foreground_greater = np.greater(foreground_norm[:, :, :3], 0.5)
            min_element = np.minimum(base_norm[:, :, :3] * (foreground_norm[:, :, :3] * 2.0), 1.0)
            inverse_min_element = np.minimum(
                1.0 - ((1.0 - base_norm[:, :, :3]) * (1.0 - (foreground_norm[:, :, :3] - 0.5) * 2.0)),
                1.0,
            )
            comp_value = (base_greater * inverse_min_element) + (np.logical_not(foreground_greater) * min_element)

        elif self.overlay_types == "grain_extract":
            comp_value = np.clip(base_norm[:, :, :3] - foreground_norm[:, :, :3] + 0.5, 0.0, 1.0)

        elif self.overlay_types == "grain_merge":
            comp_value = np.clip(base_norm[:, :, :3] + foreground_norm[:, :, :3] - 0.5, 0.0, 1.0)

        elif self.overlay_types == "overlay":
            base_less = np.less(base_norm[:, :, :3], 0.5)
            base_greater_equal = np.greater_equal(base_norm[:, :, :3], 0.5)
            base_foreground_product = 2 * base_norm[:, :, :3] * foreground_norm[:, :, :3]
            inverse_base_foreground_product = 1 - (2 * (1 - base_norm[:, :, :3]) * (1 - foreground_norm[:, :, :3]))
            comp_value = (base_less * base_foreground_product) + (base_greater_equal * inverse_base_foreground_product)

        # get reshaped ratio
        ratio_rs = np.reshape(np.repeat(ratio, 3), (base_norm.shape[0], base_norm.shape[1], 3))

        # blend image
        if self.overlay_types == "addition" or self.overlay_types == "subtract":
            # clip value for addition or subtract
            img_blended = np.clip((comp_value * ratio_rs) + (base_norm * (1.0 - ratio_rs)), 0.0, 1.0)

        else:
            img_blended = (comp_value * ratio_rs) + (base_norm * (1.0 - ratio_rs))

        # get blended image in uint8
        img_blended = (img_blended * 255).astype("uint8")

        # add patch of blended image back to background
        overlay_background[ystart:yend, xstart:xend] = img_blended

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
            base_y, base_x = base.shape[:2]

            # center of overlay
            if bg_width > fg_width:
                center_x = xstart + int(fg_width / 2)
            else:
                center_x = xstart + int(bg_width / 2)
            if bg_height > fg_height:
                center_y = ystart + int(fg_height / 2)
            else:
                center_y = ystart + int(bg_height / 2)
            center = (center_x, center_y)

            # check for size mismatch issue
            new_foreground, center = self.check_size(self.foreground, overlay_background, center)

            # new foreground height and width
            fg_height, fg_width = new_foreground.shape[:2]

            # check if new foreground size > width or height
            half_width = int((xend - xstart) / 2)
            half_height = int((yend - ystart) / 2)
            foreground_half_width = int(fg_width / 2)
            foreground_half_height = int(fg_height / 2)
            if foreground_half_width > half_width:
                half_difference = foreground_half_width - half_width
                new_foreground = new_foreground[:, half_difference:-half_difference]
            if foreground_half_height > half_height:
                half_difference = foreground_half_height - half_height
                new_foreground = new_foreground[half_difference:-half_difference, :]

            # resize new_foreground to cropped background size
            if self.overlay_types != "mix":
                new_foreground = cv2.resize(new_foreground, (base_x, base_y), interpolation=cv2.INTER_AREA)

            # get new size of foreground again
            fg_height, fg_width = new_foreground.shape[:2]

            # convert foreground to gray again
            if len(new_foreground.shape) > 2:
                new_foreground_gray = cv2.cvtColor(new_foreground, cv2.COLOR_BGR2GRAY)
            else:
                new_foreground_gray = new_foreground

            # min or max overlay types
            if self.overlay_types == "min" or self.overlay_types == "max":
                self.min_max_blend(base, base_gray, new_foreground, new_foreground_gray, fg_height, fg_width)

            # mix overlay type
            elif self.overlay_types == "mix":
                overlay_background = self.mix_blend(overlay_background, new_foreground, center, fg_height, fg_width)

            # normal overlay type using alpha value
            elif self.overlay_types == "normal":
                self.normal_blend(overlay_background, base, new_foreground, xstart, xend, ystart, yend)

            # overlay types:
            # lighten, darken, addition, subtract, difference, screen, dodge
            # multiply, divide, hard_light, grain_extract, grain_merge, overlay
            else:
                self.various_blend(overlay_background, base, new_foreground, xstart, xend, ystart, yend)

            # get original height and width from foreground
            fg_height, fg_width = self.foreground.shape[:2]

            if self.edge == "left" or self.edge == "right":
                # for next loop ystart and yend
                ystart += fg_height + offset_height
                yend = ystart + fg_height
                if self.overlay_types != "mix":
                    # break when next ystart is > image y size
                    if ystart >= bg_height - fg_height:
                        break

            elif self.edge == "top" or self.edge == "bottom":
                # for next loop xstart and xend
                xstart += fg_width + offset_width
                xend = xstart + fg_width
                if self.overlay_types != "mix":
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
        random_height_scale = random.randint(min_value, max_value)
        random_width_scale = random.randint(min_value, max_value)

        new_fg_height = int((self.foreground.shape[0] * random_height_scale) / 100)
        new_fg_width = int((self.foreground.shape[1] * random_width_scale) / 100)
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
            if bg_height > fg_height:
                ystart = int(bg_height / 2) - int(fg_height / 2)
                yend = ystart + fg_height
            else:
                ystart = 0
                yend = bg_height
            if bg_width > fg_width:
                xstart = int(bg_width / 2) - int(fg_width / 2)
                xend = xstart + fg_width
            else:
                xstart = 0
                xend = bg_width

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

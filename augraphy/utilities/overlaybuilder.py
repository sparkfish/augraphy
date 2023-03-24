import math
import random

import cv2
import numpy as np
from numba import config
from numba import jit

from augraphy.augmentations.lib import make_white_transparent


class OverlayBuilder:
    """Takes an input image, a number of times to duplicate that image, an image
    on which to overlay the result of this, and a page position, then produces
    an overlayable image with the input image copied that many times across
    the edge of the page or at random location or at the center of image.

    :param overlay_types: Types of overlay method.
    :type overlay_types: string
    :param foreground: The image to overlay on the background document.
    :type foreground: numpy array
    :param background: The document.
    :type background: numpy array
    :param ntimes: Number copies of the foreground image to draw.
    :type ntimes: int, optional
    :param nscales: Scales of foreground image size.
    :type nscales: tuple, optional
    :param edge: Which edge of the page the foreground copies should be
        placed on. Selections included left, right, top, bottom, enter, random.
    :type edge: string, optional
    :param edge_offset: How far from the edge of the page to draw the copies.
    :type edge_offset: int, optional
    :param alpha: Alpha value for overlay methods that uses alpha in the blending.
    :type alpha: float, optional
    :param ink_color: Ink color value for ink_to_paper overlay type.
    :type ink_color: int, optional
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
        ink_color=-1,
    ):
        self.overlay_types = overlay_types
        self.foreground = foreground
        self.background = background
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edge_offset = max(0, edge_offset)  # prevent negative
        self.alpha = alpha
        self.ink_color = ink_color

        # set valid edge type
        if edge not in ["center", "random", "left", "right", "top", "bottom"]:
            self.edge = "center"

        # most of the blending methods are adapted here: https://github.com/flrs/blend_modes
        # set valid overlay types
        if overlay_types not in [
            "ink_to_paper",
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

    def compute_offsets(self, foreground):
        """Determine where to place the foreground image copies

        :param foreground: The image to overlay on the background document.
        :type foreground: numpy array
        """
        xdim = self.background.shape[1]
        ydim = self.background.shape[0]

        img_width = foreground.shape[1]
        img_height = foreground.shape[0]

        remaining_width = xdim - (self.ntimes * img_width)
        remaining_height = ydim - (self.ntimes * img_height)

        # max to prevent negative offset
        offset_width = max(0, math.floor(remaining_width / (self.ntimes + 1)))
        offset_height = max(0, math.floor(remaining_height / (self.ntimes + 1)))

        return offset_width, offset_height

    def check_size(self, img_foreground, img_background, center=None):
        """Check the fitting size of foreground to background

        :param img_foreground: The image to overlay on the background document.
        :type img_foreground: numpy array
        :param img_background: The background document.
        :type img_background: numpy array
        :param center: Center coordinate (x,y) of the overlaying process.
        :type center: tuple
        """

        # background size
        ysize_background, xsize_background = img_background.shape[:2]

        # get center x and y
        center_x, center_y = center

        # foreground size
        ysize_foreground, xsize_foreground = img_foreground.shape[:2]

        # center point of foreground
        ysize_half_foreground, xsize_half_foreground = int(ysize_foreground / 2), int(
            xsize_foreground / 2,
        )

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
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )

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
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )

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
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )
            # update new center
            center = [xsize_background - xsize_half_foreground, center[1]]

        # if foreground x < 0
        if center_x - xsize_half_foreground < 0:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[:, abs(center_x - xsize_half_foreground) :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )
            # update new center
            center = [xsize_half_foreground, center[1]]

        # if foreground y exceed background height
        if center_y + ysize_half_foreground > ysize_background:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[: -(center_y + ysize_half_foreground - ysize_background), :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )

            # update new center
            center = [center[0], ysize_background - ysize_half_foreground]

        # if foreground y < 0
        if center_y - ysize_half_foreground < 0:

            # get new patch image to not exceed background width
            img_foreground = img_foreground[abs(center_y - ysize_half_foreground) :, :]
            # get new foreground size
            ysize_foreground, xsize_foreground = img_foreground.shape[:2]
            # half new foreground size
            ysize_half_foreground, xsize_half_foreground = (
                int(
                    ysize_foreground / 2,
                ),
                int(xsize_foreground / 2),
            )
            # update new center
            center = [center[0], ysize_half_foreground]

        return img_foreground, center

    @staticmethod
    @jit(nopython=True, cache=True)
    def compose_alpha(img_alpha_background, img_alpha_foreground, alpha):
        """Calculate alpha composition ratio between two images.

        :param img_alpha_background: The background image alpha layer.
        :type img_alpha_background: numpy array
        :param img_alpha_foreground: The foreground image alpha layer.
        :type img_alpha_foreground: numpy array
        :param alpha: Alpha value for the blending process.
        :type alpha: float
        """

        comp_alpha = np.minimum(img_alpha_background, img_alpha_foreground) * alpha
        new_alpha = img_alpha_background + (1.0 - img_alpha_foreground) * comp_alpha
        ratio = comp_alpha / new_alpha

        return ratio

    def ink_to_paper_blend(
        self,
        overlay_background,
        base,
        new_foreground,
        xstart,
        xend,
        ystart,
        yend,
    ):
        """Apply blending using default ink to paper printing method.

        :param overlay_background: Background image.
        :type overlay_background: numpy array
        :param base: A patch of background image.
        :type base: numpy array
        :param new_foreground: Foreground_image.
        :type new_foreground: numpy array
        :param xstart: x start point of the image patch.
        :type xstart: int
        :param xend: x end point of the image patch.
        :type xend: int
        :param ystart: y start point of the image patch.
        :type ystart: int
        :param yend: y end point of the image patch.
        :type yend: int
        """

        foreground = make_white_transparent(new_foreground, self.ink_color)
        # Split out the transparency mask from the colour info
        overlay_img = foreground[:, :, :3]  # Grab the BRG planes
        overlay_mask = foreground[:, :, 3:]  # And the alpha plane

        # Turn the single channel alpha masks into three channel, so we can use them as weights
        if len(foreground.shape) > 2:
            overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Convert background to 3 channels if they are in single channel
        if len(base.shape) < 3:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        background_part = (base * (1.0 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1.0 / 255.0)) * (overlay_mask * (1.0 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        image_printed = np.uint8(
            cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0),
        )

        overlay_background[ystart:yend, xstart:xend] = image_printed

        return overlay_background

    def mix_blend(
        self,
        overlay_background,
        new_foreground,
        center,
        fg_height,
        fg_width,
    ):
        """Apply blending using cv2.seamlessClone.

        :param overlay_background: The background image.
        :type overlay_background: numpy array
        :param new_foreground: The foreground iamge of overlaying process.
        :type new_foreground: numpy array
        :param center: Center coordinate (x,y) of the overlaying process.
        :type center: tuple
        :param fg_height: Height of foreground image.
        :type fg_height: int
        :param fg_width: Width of foreground image.
        :type fg_width: int
        """

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

    def min_max_blend(
        self,
        base,
        base_gray,
        new_foreground,
        new_foreground_gray,
        fg_height,
        fg_width,
    ):
        """Apply blending using min or max gray value.

        :param base: Background image.
        :type base: numpy array
        :param base_gray: Background image in grayscale.
        :type base_gray: numpy array
        :param new_foreground: Foreground_image.
        :type new_foreground: numpy array
        :param new_foreground_gray: Foreground image in grayscale.
        :type new_foreground_gray: numpy array
        :param fg_height: Height of foreground image.
        :type fg_height: int
        :param fg_width: Width of foreground image.
        :type fg_width: int
        """

        if self.overlay_types == "min":
            indices = new_foreground_gray < base_gray
        else:
            indices = new_foreground_gray > base_gray

        # for colour image
        if len(base.shape) > 2:
            for i in range(base.shape[2]):
                base[:, :, i][indices] = new_foreground[:, :, i][indices]
        # for grayscale
        else:
            base[indices] = new_foreground_gray[indices]

    def normal_blend(
        self,
        overlay_background,
        base,
        new_foreground,
        xstart,
        xend,
        ystart,
        yend,
    ):
        """Apply blending using input alpha value (normal method).

        :param overlay_background: Background image.
        :type overlay_background: numpy array
        :param base: A patch of background image.
        :type base: numpy array
        :param new_foreground: Foreground_image.
        :type new_foreground: numpy array
        :param xstart: x start point of the image patch.
        :type xstart: int
        :param xend: x end point of the image patch.
        :type xend: int
        :param ystart: y start point of the image patch.
        :type ystart: int
        :param yend: y end point of the image patch.
        :type yend: int
        """

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

    def various_blend(
        self,
        overlay_background,
        base,
        new_foreground,
        xstart,
        xend,
        ystart,
        yend,
    ):
        """Apply blending using input alpha value (multiple methods).

        :param overlay_background: Background image.
        :type overlay_background: numpy array
        :param base: A patch of background image.
        :type base: numpy array
        :param new_foreground: Foreground_image.
        :type new_foreground: numpy array
        :param xstart: x start point of the image patch.
        :type xstart: int
        :param xend: x end point of the image patch.
        :type xend: int
        :param ystart: y start point of the image patch.
        :type ystart: int
        :param yend: y end point of the image patch.
        :type yend: int
        """

        # convert to float (0-1)
        base_norm = base / 255.0
        foreground_norm = new_foreground / 255.0

        check_alpha_ratio = 0
        # get alpha layer (if any)
        if len(base_norm.shape) > 3 and len(foreground_norm.shape) > 3:
            img_base_alpha = base_norm[:, :, 3]
            img_foreground_alpha = foreground_norm[:, :, 3]
            check_alpha_ratio = 1

            # compose alpha ratio from background and foreground alpha value
            ratio = self.compose_alpha(img_base_alpha, img_foreground_alpha, self.alpha)
            # remove infinity value due to zero division
            ratio[ratio == np.inf] = 0

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
            comp_value = np.minimum(
                base_norm[:, :, :3] / (1.0 - foreground_norm[:, :, :3]),
                1.0,
            )

        elif self.overlay_types == "multiply":
            comp_value = np.clip(
                base_norm[:, :, :3] * foreground_norm[:, :, :3],
                0.0,
                1.0,
            )

        elif self.overlay_types == "divide":
            comp_value = np.minimum(
                (256.0 / 255.0 * base_norm[:, :, :3]) / (1.0 / 255.0 + foreground_norm[:, :, :3]),
                1.0,
            )

        elif self.overlay_types == "hard_light":
            base_greater = np.greater(base_norm[:, :, :3], 0.5)
            foreground_greater = np.greater(foreground_norm[:, :, :3], 0.5)
            min_element = np.minimum(
                base_norm[:, :, :3] * (foreground_norm[:, :, :3] * 2.0),
                1.0,
            )
            inverse_min_element = np.minimum(
                1.0 - ((1.0 - base_norm[:, :, :3]) * (1.0 - (foreground_norm[:, :, :3] - 0.5) * 2.0)),
                1.0,
            )
            comp_value = (base_greater * inverse_min_element) + (np.logical_not(foreground_greater) * min_element)

        elif self.overlay_types == "grain_extract":
            comp_value = np.clip(
                base_norm[:, :, :3] - foreground_norm[:, :, :3] + 0.5,
                0.0,
                1.0,
            )

        elif self.overlay_types == "grain_merge":
            comp_value = np.clip(
                base_norm[:, :, :3] + foreground_norm[:, :, :3] - 0.5,
                0.0,
                1.0,
            )

        elif self.overlay_types == "overlay":
            base_less = np.less(base_norm[:, :, :3], 0.5)
            base_greater_equal = np.greater_equal(base_norm[:, :, :3], 0.5)
            base_foreground_product = 2 * base_norm[:, :, :3] * foreground_norm[:, :, :3]
            inverse_base_foreground_product = 1 - (2 * (1 - base_norm[:, :, :3]) * (1 - foreground_norm[:, :, :3]))
            comp_value = (base_less * base_foreground_product) + (base_greater_equal * inverse_base_foreground_product)

        # apply alpha ratio only if both images have alpha layer
        if check_alpha_ratio:
            # get reshaped ratio
            ratio_rs = np.reshape(
                np.repeat(ratio, 3),
                (base_norm.shape[0], base_norm.shape[1], 3),
            )

            # blend image
            if self.overlay_types == "addition" or self.overlay_types == "subtract":
                # clip value for addition or subtract
                img_blended = np.clip(
                    (comp_value * ratio_rs) + (base_norm * (1.0 - ratio_rs)),
                    0.0,
                    1.0,
                )
            else:
                img_blended = self.apply_ratio(comp_value, base_norm, ratio_rs)
        else:
            img_blended = comp_value

        # get blended image in uint8
        img_blended = (img_blended * 255).astype("uint8")

        # add patch of blended image back to background
        overlay_background[ystart:yend, xstart:xend] = img_blended

    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_ratio(comp_value, base_norm, ratio_rs):
        """Function to apply alpha ratio to both foreground and background image

        :param comp_value: The resulting image from blending process.
        :type comP_value: numpy array
        :param base_norm: The background.
        :type base_norm: numpy array
        :param ratio_rs: Alpha ratio for each pixel.
        :type ratio_rs: numpy array
        """

        return (comp_value * ratio_rs) + (base_norm * (1.0 - ratio_rs))

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
        """Applies overlay from foreground to background.

        :param overlay_background: Background image.
        :type overlay_background: numpy array
        :param offset_width: Offset width value to the overlay process.
        :type offset_width: int
        :param offset_height: Offset height value to the overlay process.
        :type offset_height: int
        :param ystart: y start point of the overlaying process.
        :type ystart: int
        :param yend: y end point of the overlaying process.
        :type yend: int
        :param xstart: x start point of the overlaying process.
        :type xstart: int
        :param xend: x end point of the overlaying process.
        :type xend: int
        """

        # get bgr and gray of background
        if len(overlay_background.shape) > 2:
            overlay_background_gray = cv2.cvtColor(
                overlay_background,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            overlay_background_gray = overlay_background
            overlay_background = cv2.cvtColor(overlay_background, cv2.COLOR_GRAY2BGR)

        if isinstance(self.foreground, list):

            for i, current_foreground in enumerate(self.foreground):
                # get bgr and gray of foreground
                if len(current_foreground.shape) < 3:
                    self.foreground[i] = cv2.cvtColor(
                        current_foreground,
                        cv2.COLOR_GRAY2BGR,
                    )
            fg_height, fg_width = self.foreground[0].shape[:2]
        else:
            # get bgr and gray of foreground
            if len(self.foreground.shape) < 3:
                self.foreground = cv2.cvtColor(self.foreground, cv2.COLOR_GRAY2BGR)
            fg_height, fg_width = self.foreground.shape[:2]

        # get size
        bg_height, bg_width = overlay_background.shape[:2]
        for i in range(self.ntimes):

            if isinstance(self.foreground, list):
                foreground = random.choice(self.foreground)
            else:
                foreground = self.foreground

            if self.edge == "random":
                ystart = random.randint(0, bg_height - 10)
                yend = ystart + fg_height
                xstart = random.randint(0, bg_width - 10)
                xend = xstart + fg_width

            # prevent negative value
            ystart = max(0, ystart)
            xstart = max(0, xstart)

            # prevent 0 size
            if yend - ystart == 0:
                yend += 1
            if xend - xstart == 0:
                xend += 1

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
            new_foreground, center = self.check_size(
                foreground,
                overlay_background,
                center,
            )

            # new foreground height and width
            fg_height, fg_width = new_foreground.shape[:2]

            # check if new foreground size > width or height
            half_width = int((xend - xstart) / 2)
            half_height = int((yend - ystart) / 2)
            foreground_half_width = int(fg_width / 2)
            foreground_half_height = int(fg_height / 2)
            if foreground_half_width > half_width:
                half_difference = foreground_half_width - half_width
                # remove right part
                if self.edge == "left":
                    new_foreground = new_foreground[:, : -half_difference * 2]
                # remove left part
                elif self.edge == "right":
                    new_foreground = new_foreground[:, half_difference * 2 :]
                # shift evenly
                else:
                    new_foreground = new_foreground[:, half_difference:-half_difference]
            if foreground_half_height > half_height:
                half_difference = foreground_half_height - half_height
                # remove top part
                if self.edge == "bottom":
                    new_foreground = new_foreground[half_difference * 2 :, :]
                # remove bottom part
                elif self.edge == "top":
                    new_foreground = new_foreground[: -half_difference * 2, :]
                # shift evenly
                else:
                    new_foreground = new_foreground[half_difference:-half_difference, :]

            # resize new_foreground to cropped background size
            if self.overlay_types != "mix":
                new_foreground = cv2.resize(
                    new_foreground,
                    (base_x, base_y),
                    interpolation=cv2.INTER_AREA,
                )

            # get new size of foreground again
            fg_height, fg_width = new_foreground.shape[:2]

            # convert foreground to gray again
            if len(new_foreground.shape) > 2:
                new_foreground_gray = cv2.cvtColor(new_foreground, cv2.COLOR_BGR2GRAY)
            else:
                new_foreground_gray = new_foreground

            # ink to paper overlay type
            if self.overlay_types == "ink_to_paper":
                self.ink_to_paper_blend(
                    overlay_background,
                    base,
                    new_foreground,
                    xstart,
                    xend,
                    ystart,
                    yend,
                )

            # min or max overlay types
            elif self.overlay_types == "min" or self.overlay_types == "max":
                self.min_max_blend(
                    base,
                    base_gray,
                    new_foreground,
                    new_foreground_gray,
                    fg_height,
                    fg_width,
                )

            # mix overlay type
            elif self.overlay_types == "mix":
                overlay_background = self.mix_blend(
                    overlay_background,
                    new_foreground,
                    center,
                    fg_height,
                    fg_width,
                )

            # normal overlay type using alpha value
            elif self.overlay_types == "normal":
                self.normal_blend(
                    overlay_background,
                    base,
                    new_foreground,
                    xstart,
                    xend,
                    ystart,
                    yend,
                )

            # overlay types:
            # lighten, darken, addition, subtract, difference, screen, dodge
            # multiply, divide, hard_light, grain_extract, grain_merge, overlay
            else:
                self.various_blend(
                    overlay_background,
                    base,
                    new_foreground,
                    xstart,
                    xend,
                    ystart,
                    yend,
                )

            # get original height and width from foreground
            fg_height, fg_width = foreground.shape[:2]

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

        random_height_scale = np.random.uniform(self.nscales[0], self.nscales[1])
        random_width_scale = np.random.uniform(self.nscales[0], self.nscales[1])

        if isinstance(self.foreground, list):

            new_fg_height = int((self.foreground[0].shape[0] * random_height_scale))
            new_fg_width = int((self.foreground[0].shape[1] * random_width_scale))

            for i, current_foreground in enumerate(self.foreground):
                self.foreground[i] = cv2.resize(
                    current_foreground,
                    (int(new_fg_width), int(new_fg_height)),
                    interpolation=cv2.INTER_AREA,
                )
            foreground = self.foreground[0]
        else:
            foreground = self.foreground
            new_fg_height = int((foreground.shape[0] * random_height_scale))
            new_fg_width = int((foreground.shape[1] * random_width_scale))

        # foreground size (height & width)
        fg_height, fg_width = foreground.shape[:2]

        # background size (height & width)
        bg_height, bg_width = self.background.shape[:2]

        # compute offsets between foreground and background
        offset_width, offset_height = self.compute_offsets(foreground)

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

        # convert image back to gray if background image is in grayscale
        if len(self.background.shape) < 3 and len(overlay_background.shape) > 2:
            overlay_background = cv2.cvtColor(overlay_background, cv2.COLOR_BGR2GRAY)

        return overlay_background

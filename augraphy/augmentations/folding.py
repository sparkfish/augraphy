import random

import numpy as np

from augraphy.augmentations.lib import rotate_bounding_boxes
from augraphy.augmentations.lib import rotate_image_PIL
from augraphy.augmentations.lib import rotate_keypoints
from augraphy.augmentations.lib import update_mask_labels
from augraphy.augmentations.lib import warp_fold
from augraphy.base.augmentation import Augmentation


class Folding(Augmentation):
    """Emulates folding effect from perspective transformation

    :param fold_x: X coordinate of the folding effect.
    :type fold_x: int, optional
    :param fold_deviation: Deviation (in pixels) of provided X coordinate location.
    :type fold_deviation: tuple, optional
    :param fold count: Number of applied foldings
    :type fold_count: int, optional
    :param fold_noise: Level of noise added to folding area. Range from
        value of 0 to 1.
    :type fold_noise: float, optional
    :param fold_angle_range: Tuple of ints determining the angle to rotate the image
        before applying a varying angle folding effect.
    :type fold_angle_range: tuple, optional
    :param gradient_width: Tuple (min, max) Measure of the space affected
        by fold prior to being warped (in units of percentage of width of page).
    :type gradient_width: tuple, optional
    :param gradient_height: Tuple (min, max) Measure of depth of fold (unit
        measured as percentage page height)
    :type gradient_height: tuple, optional
    :param backdrop_color: The backdrop color (BGR) of the folding effect.
    :type backdrop_color: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        fold_x=None,
        fold_deviation=(0, 0),
        fold_count=2,
        fold_noise=0.01,
        fold_angle_range=(0, 0),
        gradient_width=(0.1, 0.2),
        gradient_height=(0.01, 0.02),
        backdrop_color=(0, 0, 0),
        p=1,
    ):
        super().__init__(p=p)
        self.fold_x = fold_x
        self.fold_deviation = fold_deviation
        self.fold_count = fold_count
        self.fold_noise = fold_noise
        self.fold_angle_range = fold_angle_range
        self.gradient_width = gradient_width
        self.gradient_height = gradient_height
        self.backdrop_color = backdrop_color

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Folding(fold_x={self.fold_x}, fold_deviation={self.fold_deviation}, fold_count={self.fold_count}, fold_noise={self.fold_noise}, fold_angle_range={self.fold_angle_range}, gradient_width={self.gradient_width}, gradient_height={self.gradient_height}, backdrop_color={self.backdrop_color}, p={self.p})"

    def apply_folding(
        self,
        img,
        keypoints,
        bounding_boxes,
        ysize,
        xsize,
        fold_x,
        fold_width_one_side,
        fold_y_shift,
        fold_noise,
        fmask,
    ):
        """Apply perspective transform twice to get single folding effect.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        :param ysize: Height of the image.
        :type ysize: int
        :param xsize: Width of the image.
        :type xsize: int
        :param gradient_width:  Measure of the space affected by fold prior to being warped (in units of percentage of width of page).
        :type gradient_width: int
        :param gradient_height: Measure of depth of fold (unit measured as percentage page height).
        :type gradient_height: int
        :param fold_noise: Level of noise added to folding area.
        :type fold_noise: float
        :param fmask: Flag to identify if the input is mask instead of image.
        :type fmask: int
        """

        # test for valid folding center line
        if (xsize - fold_width_one_side - 1) < (fold_width_one_side + 1):
            print("Folding augmentation is not applied, please increase image size")
            return img

        if (fold_width_one_side != 0) and (fold_y_shift != 0):
            img_fold_l = warp_fold(
                img,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
                side="left",
                backdrop_color=self.backdrop_color,
                fmask=fmask,
            )
            img_fold_r = warp_fold(
                img_fold_l,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
                side="right",
                backdrop_color=self.backdrop_color,
                fmask=fmask,
            )

            if not fmask:
                # warp keypoints
                if keypoints is not None:
                    lx0, ly0, lxn, lyn = fold_x - fold_width_one_side, 0, fold_x, ysize
                    rx0, ry0, rxn, ryn = fold_x, 0, fold_x + (fold_width_one_side), ysize

                    # y shifting value for single pixel
                    y_shift_single_step = fold_y_shift / fold_width_one_side

                    # warp each label
                    for name, points in keypoints.items():
                        for i, (xpoint, ypoint) in enumerate(points):
                            # test for left box
                            if xpoint >= lx0 and xpoint < lxn and ypoint >= ly0 and ypoint < lyn:
                                # scale y shift based on their distance to center x of folding
                                ypoint += round((xpoint - lx0) * y_shift_single_step)
                            # test for right box
                            elif xpoint >= rx0 and xpoint < rxn and ypoint >= ry0 and ypoint < ryn:
                                # scale y shift based on their distance to center x of folding
                                ypoint += round((fold_width_one_side - (xpoint - fold_x)) * y_shift_single_step)
                            points[i] = [xpoint, ypoint]

                # warp bounding boxes
                if bounding_boxes is not None:
                    lx0, ly0, lxn, lyn = fold_x - fold_width_one_side, 0, fold_x, ysize
                    rx0, ry0, rxn, ryn = fold_x, 0, fold_x + (fold_width_one_side), ysize

                    # y shifting value for single pixel
                    y_shift_single_step = fold_y_shift / fold_width_one_side

                    # warp each box
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        width = xepoint - xspoint
                        height = yepoint - yspoint
                        # test for left box
                        if xspoint >= lx0 and xspoint < lxn and yspoint >= ly0 and yspoint < lyn:
                            # scale y shift based on their distance to center x of folding
                            yspoint += round((xspoint - lx0) * y_shift_single_step)
                        # test for right box
                        elif xspoint >= rx0 and xspoint < rxn and yspoint >= ry0 and yspoint < ryn:
                            # scale y shift based on their distance to center x of folding
                            yspoint += round((fold_width_one_side - (xspoint - fold_x)) * y_shift_single_step)
                        bounding_boxes[i] = [xspoint, yspoint, xspoint + width, yspoint + height]

            return img_fold_r

        else:
            if fold_width_one_side == 0:
                print(
                    "Folding augmentation is not applied, please increase gradient width or image size",
                )
            else:
                print(
                    "Folding augmentation is not applied, please increase gradient height or image size",
                )
            return img

    def apply_rotate_and_folding(
        self,
        image_fold,
        fold_angle,
        fold_x=None,
        fold_width_one_side=None,
        fold_y_shift=None,
        keypoints=None,
        bounding_boxes=None,
        fmask=0,
    ):
        """Apply rotation and folding effect.

        :param image_fold: The image to apply the function.
        :type image_gold: numpy.array (numpy.uint8)
        :param fold_angle: The angle of rotation.
        :type fold_angle: int
        :param fold_x: The folding center x coordinate
        :type fold_x: int
        :param fold_width_one_side: The warped width of folding effect from the fold_x.
        :type fold_width_one_side: int
        :param fold_y_shift: Depth of the folding effect.
        :type fold_y_shift: int
        :param fmask: Flag to identify if the input image is mask isntead of image.
        :type fmask: int
        """

        # input image size
        iysize, ixsize = image_fold.shape[:2]

        if fold_angle != 0:
            # rotate image before the folding
            image_fold = rotate_image_PIL(
                image_fold,
                angle=fold_angle,
                background_value=self.backdrop_color,
                expand=1,
            )

            if not fmask:
                # rotate keypoints
                if keypoints is not None:
                    # center of rotation
                    cy = int(iysize / 2)
                    cx = int(ixsize / 2)
                    # compute offset after rotation
                    rysize, rxsize = image_fold.shape[:2]
                    y_offset = (rysize / 2) - cy
                    x_offset = (rxsize / 2) - cx
                    # rotate keypoints
                    rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, -fold_angle)

                # rotate bounding boxes
                if bounding_boxes is not None:
                    # center of rotation
                    cy = int(iysize / 2)
                    cx = int(ixsize / 2)
                    # compute offset after rotation
                    rysize, rxsize = image_fold.shape[:2]
                    y_offset = (rysize / 2) - cy
                    x_offset = (rxsize / 2) - cx
                    # rotate bounding boxes
                    # use - fold_angle because image are rotated anticlockwise
                    rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, -fold_angle)

        # rotated size
        ysize, xsize = image_fold.shape[:2]

        # create folding parameters (for non mask, mask will be using input parameters)
        if not fmask:

            # folding width from left to center of folding, or from right to center of folding
            min_fold_x = min(np.ceil(self.gradient_width[0] * xsize), xsize).astype("int")
            max_fold_x = min(np.ceil(self.gradient_width[1] * xsize), xsize).astype("int")
            fold_width_one_side = int(
                random.randint(min_fold_x, max_fold_x) / 2,
            )

            # center of folding
            if self.fold_x is None:
                fold_x = random.randint(
                    fold_width_one_side + 1,
                    xsize - fold_width_one_side - 1,
                )
            else:
                deviation = random.randint(
                    self.fold_deviation[0],
                    self.fold_deviation[1],
                ) * random.choice([-1, 1])
                fold_x = min(
                    max(self.fold_x + deviation, fold_width_one_side + 1),
                    xsize - fold_width_one_side - 1,
                )

            # y distortion in folding (support positive y value for now)
            fold_y_shift_min = min(np.ceil(self.gradient_height[0] * ysize), ysize).astype("int")
            fold_y_shift_max = min(np.ceil(self.gradient_height[1] * ysize), ysize).astype("int")
            fold_y_shift = random.randint(
                fold_y_shift_min,
                fold_y_shift_max,
            )

        # no noise for mask
        if fmask:
            fold_noise = 0
        else:
            fold_noise = self.fold_noise

        image_fold = self.apply_folding(
            image_fold,
            keypoints,
            bounding_boxes,
            image_fold.shape[0],
            image_fold.shape[1],
            fold_x,
            fold_width_one_side,
            fold_y_shift,
            fold_noise,
            fmask,
        )

        if fold_angle != 0:
            # rotate back the image
            image_fold = rotate_image_PIL(
                image_fold,
                angle=-fold_angle,
                background_value=self.backdrop_color,
                expand=1,
            )

            # rotate keypoints
            if not fmask:
                if keypoints is not None:
                    # center of rotation
                    cy = int(ysize / 2)
                    cx = int(xsize / 2)
                    # compute offset after rotation
                    rysize, rxsize = image_fold.shape[:2]
                    y_offset = (rysize / 2) - cy
                    x_offset = (rxsize / 2) - cx
                    # rotate keypoints
                    rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, fold_angle)

                # rotate bounding boxes
                if bounding_boxes is not None:
                    # center of rotation
                    cy = int(ysize / 2)
                    cx = int(xsize / 2)
                    # compute offset after rotation
                    rysize, rxsize = image_fold.shape[:2]
                    y_offset = (rysize / 2) - cy
                    x_offset = (rxsize / 2) - cx
                    # rotate bounding boxes
                    # use - fold_angle because image are rotated anticlockwise
                    rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, fold_angle)

            # get the image without the padding area, we will get extra padding area after the rotation
            rysize, rxsize = image_fold.shape[:2]

            # center of x and y
            cx = int(rxsize / 2)
            cy = int(rysize / 2)
            rcx = int(ixsize / 2)
            rcy = int(iysize / 2)
            start_x = cx - rcx
            start_y = cy - rcy
            end_x = start_x + ixsize
            end_y = start_y + iysize
            image_fold = image_fold[start_y:end_y, start_x:end_x]

            # remove padding area of keypoints and bounding boxes
            if not fmask:
                if keypoints is not None:
                    y_offset = (iysize / 2) - (rysize / 2)
                    x_offset = (ixsize / 2) - (rxsize / 2)
                    for name, points in keypoints.items():
                        for i, (xpoint, ypoint) in enumerate(points):
                            points[i] = [round(xpoint + x_offset), round(ypoint + y_offset)]

                if bounding_boxes is not None:
                    y_offset = (iysize / 2) - (rysize / 2)
                    x_offset = (ixsize / 2) - (rxsize / 2)
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        bounding_boxes[i] = [
                            round(xspoint + x_offset),
                            round(yspoint + y_offset),
                            round(xepoint + x_offset),
                            round(yepoint + y_offset),
                        ]

        return image_fold, fold_x, fold_width_one_side, fold_y_shift

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            # get mask unique labels
            if mask is not None:
                mask_labels = np.unique(mask).tolist() + [0]

            # get image dimension
            ysize, xsize = image.shape[:2]

            # apply folding multiple times
            image_fold = image.copy()
            for _ in range(self.fold_count):
                # random fold angle
                fold_angle = random.randint(self.fold_angle_range[0], self.fold_angle_range[1])

                # apply folding to image
                image_fold, fold_x, fold_width_one_side, fold_y_shift = self.apply_rotate_and_folding(
                    image_fold,
                    fold_angle,
                    keypoints=keypoints,
                    bounding_boxes=bounding_boxes,
                    fmask=0,
                )

                # apply folding to mask, using a same folding parameter of image
                if mask is not None:
                    mask, _, _, _ = self.apply_rotate_and_folding(
                        mask,
                        fold_angle,
                        fold_x,
                        fold_width_one_side,
                        fold_y_shift,
                        fmask=1,
                    )

            # update new interpolated mask values into each mask labels
            if mask is not None:
                update_mask_labels(mask, mask_labels)

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_fold] + outputs_extra
            else:
                return image_fold

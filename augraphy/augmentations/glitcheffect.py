import random

import cv2
import numpy as np

from augraphy.augmentations.colorshift import ColorShift
from augraphy.augmentations.lib import rotate_bounding_boxes
from augraphy.augmentations.lib import rotate_keypoints
from augraphy.augmentations.lib import update_mask_labels
from augraphy.base.augmentation import Augmentation


class GlitchEffect(Augmentation):
    """Create glitch effect by applying ColorShift and shifts patches of image horizontally or vertically.

    :param glitch_direction: Direction of the glitch effect, select from "vertical", "horizontal", "all" or "random".
    :type glitch_direction: string, optional
    :param glitch_number_range: Tuple of ints determing the number of shifted image patches.
    :type glitch_number_range: tuple, optional
    :param glitch_size_range: Tuple of ints/floats determing the size of image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image height:
            size (int) = image height  * size (float and 0.0 - 1.0)
    :type glitch_size_range: tuple, optional
    :param glitch_offset_range: Tuple of ints/floats determing the offset value to shift the image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image width:
            offset (int) = image width  * offset (float and 0.0 - 1.0)
    :type glitch_offset_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        glitch_direction="random",
        glitch_number_range=(8, 16),
        glitch_size_range=(5, 50),
        glitch_offset_range=(10, 50),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.glitch_direction = glitch_direction
        self.glitch_number_range = glitch_number_range
        self.glitch_size_range = glitch_size_range
        self.glitch_offset_range = glitch_offset_range

    def __repr__(self):
        return f"GlitchEffect(glitch_direction={self.glitch_direction}, glitch_number_range={self.glitch_number_range}, glitch_size_range={self.glitch_size_range}, glitch_offset_range={self.glitch_offset_range}, p={self.p})"

    def apply_glitch(self, image, glitch_direction, mask, keypoints, bounding_boxes):
        """Apply glitch effect into the image by shifting patches of images.

        :param image: Image to apply the glitch effect.
        :type image: numpy array
        :param direction: The direction of glitch effect.
        :type direction: string
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # input image shape
        ysize, xsize = image.shape[:2]

        # rotate image, mask, keypoints and bounding boxes for vertical direction
        if glitch_direction == "vertical":
            # rotate image
            image = np.rot90(image, 1)
            # rotate mask
            if mask is not None:
                mask_labels = np.unique(mask).tolist() + [0]
                mask = np.rot90(mask, 1)
            # rotate keypoints
            if keypoints is not None:
                # center of rotation
                cy = int(ysize / 2)
                cx = int(xsize / 2)
                # compute offset after rotation
                rysize, rxsize = image.shape[:2]
                y_offset = (rysize / 2) - cy
                x_offset = (rxsize / 2) - cx
                # apply rotation
                # use -angle because image are rotated anticlockwise
                rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, -90)
            # rotate bounding boxes
            if bounding_boxes is not None:
                # center of rotation
                cy = int(ysize / 2)
                cx = int(xsize / 2)
                # compute offset after rotation
                rysize, rxsize = image.shape[:2]
                y_offset = (rysize / 2) - cy
                x_offset = (rxsize / 2) - cx
                # use -angle because image are rotated anticlockwise
                rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, -90)

        ysize, xsize = image.shape[:2]
        glitch_number = random.randint(self.glitch_number_range[0], self.glitch_number_range[1])
        for i in range(glitch_number):

            # generate random glitch size
            if self.glitch_size_range[0] <= 1.0 and isinstance(self.glitch_size_range[0], float):
                glitch_size = random.randint(
                    int(self.glitch_size_range[0] * ysize),
                    int(self.glitch_size_range[1] * ysize),
                )
            else:
                glitch_size = random.randint(self.glitch_size_range[0], self.glitch_size_range[1])

            # generate random direction
            direction = random.choice([1, -1])

            # generate random glitch offset
            if self.glitch_offset_range[0] <= 1.0 and isinstance(self.glitch_offset_range[0], float):
                glitch_offset = (
                    random.randint(int(self.glitch_offset_range[0] * xsize), int(self.glitch_offset_range[1] * xsize))
                    * direction
                )
            else:
                glitch_offset = random.randint(self.glitch_offset_range[0], self.glitch_offset_range[1]) * direction

            # get a patch of image
            start_y = random.randint(0, ysize - glitch_size)
            image_patch = image[start_y : start_y + glitch_size, :]
            if mask is not None:
                mask_patch = mask[start_y : start_y + glitch_size, :]
            pysize, pxsize = image_patch.shape[:2]

            # create translation matrix in horizontal direction
            translation_matrix = np.float32([[1, 0, glitch_offset], [0, 1, 0]])

            # get a copy of translated area
            if direction > 0:
                image_patch_fill = image_patch[:, -glitch_offset:].copy()
            else:
                image_patch_fill = image_patch[:, glitch_offset:].copy()

            # translate image
            image_patch = cv2.warpAffine(image_patch, translation_matrix, (pxsize, pysize))
            # translate mask
            if mask is not None:
                mask_patch = cv2.warpAffine(mask_patch, translation_matrix, (pxsize, pysize))

            # translate keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        if (ypoint >= start_y) and (ypoint < (start_y + glitch_size)):
                            points[i] = [xpoint + glitch_offset, ypoint]

            # translate bounding boxes
            if bounding_boxes is not None:
                new_boxes = []
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    # both start and end point within translated area
                    if (
                        (yspoint >= start_y)
                        and (yspoint < (start_y + glitch_size))
                        and (yepoint >= start_y)
                        and (yepoint < (start_y + glitch_size))
                    ):
                        bounding_boxes[i] = [
                            max(0, xspoint + glitch_offset),
                            yspoint,
                            min(xepoint + glitch_offset, xsize - 1),
                            yepoint,
                        ]

                    # top portion of box is in translation area, but bottom portion is not
                    elif (
                        (yspoint >= start_y)
                        and (yspoint < (start_y + glitch_size))
                        and ((yepoint < start_y) or (yepoint >= (start_y + glitch_size)))
                    ):
                        # shift top box
                        bounding_boxes[i] = [
                            max(0, xspoint + glitch_offset),
                            yspoint,
                            min(xepoint + glitch_offset, xsize - 1),
                            start_y + glitch_size,
                        ]
                        # remain bottom box
                        new_boxes.append(
                            [
                                xpoint,
                                start_y + glitch_size,
                                xepoint,
                                yepoint,
                            ],
                        )

                    # bottom portion of box is in translation area, but top portion is not
                    elif (
                        ((yspoint < start_y) or (yspoint >= (start_y + glitch_size)))
                        and (yepoint >= start_y)
                        and (yepoint < (start_y + glitch_size))
                    ):
                        # shift bottom box
                        bounding_boxes[i] = [
                            max(0, xspoint + glitch_offset),
                            start_y + glitch_size,
                            min(xepoint + glitch_offset, xsize - 1),
                            yepoint,
                        ]
                        # remain top box
                        new_boxes.append(
                            [
                                xspoint,
                                yspoint,
                                xepoint,
                                start_y + glitch_size,
                            ],
                        )
                # merge boxes
                bounding_boxes += new_boxes

            # fill back the empty area after translation
            if direction > 0:
                image_patch[:, :glitch_offset] = image_patch_fill
                # mask's empty area is filled with 0
                if mask is not None:
                    mask_patch[:, :glitch_offset] = 0

            else:
                image_patch[:, glitch_offset:] = image_patch_fill
                # mask's empty area is filled with 0
                if mask is not None:
                    mask_patch[:, glitch_offset:] = 0

            # randomly scale single channel to create a single color contrast effect
            random_ratio = random.uniform(0.8, 1.2)
            channel = random.randint(0, 2)
            image_patch_ratio = image_patch[:, :, channel].astype("int") * random_ratio
            image_patch_ratio[image_patch_ratio > 255] = 255
            image_patch_ratio[image_patch_ratio < 0] = 0
            image_patch[:, :, channel] = image_patch_ratio.astype("uint8")

            image[start_y : start_y + glitch_size, :] = image_patch
            if mask is not None:
                mask[start_y : start_y + glitch_size, :] = mask_patch

        # rotate back image, mask, keypoints and bounding boxes
        if glitch_direction == "vertical":

            ysize, xsize = image.shape[:2]

            # rotate image
            image = np.rot90(image, 3)
            # rotate mask
            if mask is not None:
                mask = np.rot90(mask, 3)
                update_mask_labels(mask, mask_labels)
            # rotate keypoints
            if keypoints is not None:
                # center of rotation
                cy = int(ysize / 2)
                cx = int(xsize / 2)
                # compute offset after rotation
                rysize, rxsize = image.shape[:2]
                y_offset = (rysize / 2) - cy
                x_offset = (rxsize / 2) - cx
                # apply rotation
                # use -angle because image are rotated anticlockwise
                rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, -270)
            # rotate bounding boxes
            if bounding_boxes is not None:
                # center of rotation
                cy = int(ysize / 2)
                cx = int(xsize / 2)
                # compute offset after rotation
                rysize, rxsize = image.shape[:2]
                y_offset = (rysize / 2) - cy
                x_offset = (rxsize / 2) - cx
                # use -angle because image are rotated anticlockwise
                rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, -270)

        return image, mask

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # check and convert image into BGR format
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

            # apply color shift before the glitch effect
            color_shift = ColorShift(
                color_shift_offset_x_range=(3, 5),
                color_shift_offset_y_range=(3, 5),
                color_shift_iterations=(1, 2),
                color_shift_brightness_range=(0.9, 1.1),
                color_shift_gaussian_kernel_range=(1, 3),
                p=1,
            )
            image_output = color_shift(image)

            # check and generate random direction
            if self.glitch_direction == "random":
                glitch_direction = random.choice(["vertical", "horizontal"])
            else:
                glitch_direction = self.glitch_direction

            # for 2 directional glitches, it will be either horizontal or vertical direction first
            if glitch_direction == "all":
                horizontal_first = 0
                if random.random() > 0.5:
                    horizontal_first = 1

                # apply horizontal glitch before vertical glitch
                if horizontal_first:
                    image_output, mask = self.apply_glitch(image_output, "horizontal", mask, keypoints, bounding_boxes)

                # apply vertical glitch
                image_output, mask = self.apply_glitch(image_output, "horizontal", mask, keypoints, bounding_boxes)

                # apply horizontal glitch after vertical glitch
                if not horizontal_first:
                    image_output, mask = self.apply_glitch(image_output, "horizontal", mask, keypoints, bounding_boxes)
            else:
                image_output, mask = self.apply_glitch(image_output, glitch_direction, mask, keypoints, bounding_boxes)

            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGRA2GRAY)

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_output] + outputs_extra
            else:
                return image_output

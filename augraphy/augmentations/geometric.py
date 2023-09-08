import random

import cv2
import numpy as np

from augraphy.augmentations.lib import rotate_bounding_boxes
from augraphy.augmentations.lib import rotate_image_PIL
from augraphy.augmentations.lib import rotate_keypoints
from augraphy.augmentations.lib import update_mask_labels
from augraphy.base.augmentation import Augmentation


class Geometric(Augmentation):
    """Applies basic geometric transformations such as resizing, flips and rotation.

    :param scale: Pair of floats determining new scale of image.
    :type scale: tuple, optional
    :param translation: Pair of values determining x and y translation value.
            The translation value will be in percentage of the image size if the value is float and in between -1.0 - 1.0:
            x (int) = image width  * x (float and -1.0 - 1.0);
            y (int) = image height * y (float and -1.0 - 1.0)
    :type translation: tuple, optional
    :param fliplr: Flag to flip image in left right direction.
    :type fliplr: int, optional
    :param flipud: Flag to flip image in up down direction.
    :type flipud: int, optional
    :param crop: Tuple of 4 (x0, y0, xn, yn) to crop section of image.
             The value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
             x0 (int) = image width  * x0 (float and 0.0 - 1.0);
             y0 (int) = image height * y0 (float and 0.0 - 1.0);
             xn (int) = image width  * xn (float and 0.0 - 1.0);
             yn (int) = image height * yn (float and 0.0 - 1.0)
    :type crop: tuple, optional
    :param rotate_range: Pair of ints determining the range from which to sample
           the image rotation.
    :type rotate_range: tuple, optional
    :param randomize: Flag to apply random geometric transformations.
    :type randomize: int, optional
    :param padding: Padding amount on each (left, right, top, bottom) side.
            The padding amount will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
            left   (int) = image width  * left   (float and 0.0 - 1.0);
            right  (int) = image height * right  (float and 0.0 - 1.0);
            top    (int) = image width  * top    (float and 0.0 - 1.0);
            bottom (int) = image height * bottom (float and 0.0 - 1.0)
    :type padding: list, optional
    :param padding_type: Padding methods, select from fill,duplicate and mirror.
    :type paddng_type: string, optional
    :param padding_value: Padding value (in BGR) for fill padding method.
    :type paddng_value: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scale=(1, 1),
        translation=(0, 0),
        fliplr=0,
        flipud=0,
        crop=(),
        rotate_range=(0, 0),
        padding=[0, 0, 0, 0],
        padding_type="fill",
        padding_value=(255, 255, 255),
        randomize=0,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.scale = scale
        self.translation = translation
        self.fliplr = fliplr
        self.flipud = flipud
        self.crop = crop
        self.rotate_range = rotate_range
        self.randomize = randomize
        self.padding = padding
        self.padding_type = padding_type
        self.padding_value = padding_value

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Geometry(scale={self.scale}, translation={self.translation}, fliplr={self.fliplr}, flipud={self.flipud}, crop={self.crop}, rotate_range={self.rotate_range}, padding={self.padding}, padding_type={self.padding_type}, padding_value={self.padding_value}, randomize={self.randomize}, p={self.p})"

    def randomize_parameters(self, image):
        """Randomize parameters for random geometrical effect.

        :param image: The input image.
        :type image: numpy array
        """

        # randomize scale
        self.scale = (random.uniform(0.5, 1), random.uniform(1, 1.5))

        # randomize translation value
        ysize, xsize = image.shape[:2]
        self.translation = (random.randint(0, int(xsize * 0.1)), random.randint(0, int(ysize * 0.1)))

        # randomize flip
        self.fliplr = random.choice([0, 1])
        self.flipud = random.choice([0, 1])

        # randomize crop
        cx1 = random.randint(0, int(xsize / 5))
        cx2 = random.randint(int(xsize / 2), xsize - 1)
        cy1 = random.randint(0, int(ysize / 5))
        cy2 = random.randint(int(ysize / 2), ysize - 1)
        self.crop = (cx1, cy1, cx2, cy2)

        # randomize rotate
        self.rotate_range = (-10, 10)

        # randomize padding
        self.padding = [
            random.randint(0, int(xsize / 5)),
            random.randint(0, int(xsize / 5)),
            random.randint(0, int(ysize / 5)),
            random.randint(0, int(ysize / 5)),
        ]
        self.padding_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.padding_typee = random.choice(["fill", "mirror", "duplicate"])

    def run_crop(self, image, mask, keypoints, bounding_boxes):
        """Crop image based on the input cropping box.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # make sure there's only 4 inputs, x0, y0, xn, yn
        if len(self.crop) == 4:
            ysize, xsize = image.shape[:2]
            xstart, ystart, xend, yend = self.crop

            # when value is float and in between 0-1, scale it with image size
            if xstart >= 0 and xstart <= 1 and isinstance(xstart, float):
                xstart = int(xstart * xsize)
            if ystart >= 0 and ystart <= 1 and isinstance(ystart, float):
                ystart = int(ystart * ysize)
            if xend >= 0 and xend <= 1 and isinstance(xend, float):
                xend = int(xend * xsize)
            if yend >= 0 and yend <= 1 and isinstance(yend, float):
                yend = int(yend * ysize)

            # when value is set to -1, it takes image size
            if yend == -1:
                yend = ysize
            if xend == -1:
                xend = xsize
            # condition to make sure cropping range is valid
            check_y = yend > ystart and ystart >= 0
            check_x = xend > xstart and xstart >= 0

            if check_y and check_x:
                # crop image
                image = image[ystart:yend, xstart:xend]

                # crop mask
                if mask is not None:
                    mask = mask[ystart:yend, xstart:xend]

                # remove keypoints outside the cropping boundary
                if keypoints is not None:
                    # check each keypoint, and remove them if it is outside the cropping area
                    for name, points in keypoints.items():
                        remove_indices = []
                        # check and save the indices to be removed
                        for i, (xpoint, ypoint) in enumerate(points):
                            if xpoint < xstart or xpoint >= xend or ypoint < ystart or ypoint >= yend:
                                remove_indices.append(i)
                        # remove points
                        while remove_indices:
                            points.pop(remove_indices.pop())
                        # update points location after the cropping process
                        for i, (xpoint, ypoint) in enumerate(points):
                            xpoint -= xstart
                            ypoint -= ystart
                            points[i] = [xpoint, ypoint]

                # remove and limit bounding boxes to the cropped boundary
                if bounding_boxes is not None:
                    # check each point, and remove them if it is outside the cropping area
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        # start point is outside the croped area, but end point is inside
                        if (xspoint < xstart or xspoint >= xend or yspoint < ystart or yspoint >= yend) and (
                            xepoint >= xstart and xepoint < xend and yepoint >= ystart and yepoint < yend
                        ):
                            xspoint = min(max(xspoint, xstart), xend)
                            yspoint = min(max(yspoint, ystart), yend)
                            bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                        # end point is outside the croped area, but start point is inside
                        elif (xepoint < xstart or xepoint >= xend or yepoint < ystart or yepoint >= yend) and (
                            xspoint >= xstart and xspoint < xend and yspoint >= ystart and yspoint < yend
                        ):
                            xepoint = min(max(xepoint, xstart), xend)
                            yepoint = min(max(yepoint, ystart), yend)
                            bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                        # start point and end point are outside the croped area, remove the whole box
                        elif (xepoint < xstart or xepoint >= xend or yepoint < ystart or yepoint >= yend) and (
                            xspoint < xstart or xspoint >= xend or yspoint < ystart or yspoint >= yend
                        ):
                            remove_indices.append(i)
                    # remove boxes
                    while remove_indices:
                        bounding_boxes.pop(remove_indices.pop())
                    # update points location after the cropping process
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        xspoint -= xstart
                        yspoint -= ystart
                        xepoint -= xstart
                        yepoint -= ystart
                        bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]

        return image, mask

    def run_padding(self, image, mask, keypoints, bounding_boxes):
        """Apply padding to image based on the input padding value.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # convert from rgb to grayscale using their average
        if len(image.shape) < 3:
            padding_value = np.mean(self.padding_value)
        elif image.shape[2] == 3:
            padding_value = (self.padding_value[0], self.padding_value[1], self.padding_value[2])
        elif image.shape[2] == 4:
            # add alpha value
            padding_value = (self.padding_value[0], self.padding_value[1], self.padding_value[2], 255)

        # padding on left side
        if self.padding[0] > 0:
            # get image size
            ysize, xsize = image.shape[:2]
            # convert percentage into pixel amount
            if self.padding[0] <= 1 and isinstance(self.padding[0], float):
                self.padding[0] = int(self.padding[0] * xsize)
            # different padding shape for grayscale and colored image
            if len(image.shape) > 2:
                padding_shape = (ysize, self.padding[0], image.shape[2])
            else:
                padding_shape = (ysize, self.padding[0])
            # create the padding image
            if self.padding_type == "duplicate":
                image_padding = image[:, -self.padding[0] :].copy()
                if mask is not None:
                    mask_padding = mask[:, -self.padding[0] :].copy()
            elif self.padding_type == "mirror":
                image_padding = np.fliplr(image[:, : self.padding[0]].copy())
                if mask is not None:
                    mask_padding = np.fliplr(mask[:, : self.padding[0]].copy())
            else:
                image_padding = np.full(padding_shape, fill_value=padding_value, dtype="uint8")
                if mask is not None:
                    mask_padding = np.full(padding_shape[:2], fill_value=0, dtype="uint8")
            # combine padding image and original image
            image = np.concatenate([image_padding, image], axis=1)
            if mask is not None:
                mask = np.concatenate([mask_padding, mask], axis=1)

        # padding on right side
        if self.padding[1] > 0:
            # get image size
            ysize, xsize = image.shape[:2]
            # convert percentage into pixel amount
            if self.padding[1] <= 1 and isinstance(self.padding[1], float):
                self.padding[1] = int(self.padding[1] * xsize)
            # different padding shape for grayscale and colored image
            if len(image.shape) > 2:
                padding_shape = (ysize, self.padding[1], image.shape[2])
            else:
                padding_shape = (ysize, self.padding[1])
            # create the padding image
            if self.padding_type == "duplicate":
                image_padding = image[:, : self.padding[1]].copy()
                if mask is not None:
                    mask_padding = mask[:, : self.padding[1]].copy()
            elif self.padding_type == "mirror":
                image_padding = np.fliplr(image[:, -self.padding[1] :].copy())
                if mask is not None:
                    mask_padding = np.fliplr(mask[:, -self.padding[1] :].copy())
            else:
                image_padding = np.full(padding_shape, fill_value=padding_value, dtype="uint8")
                if mask is not None:
                    mask_padding = np.full(padding_shape[:2], fill_value=0, dtype="uint8")
            # combine padding image and original image
            image = np.concatenate([image, image_padding], axis=1)
            if mask is not None:
                mask = np.concatenate([mask, mask_padding], axis=1)

        # padding on top side
        if self.padding[2] > 0:
            # get image size
            ysize, xsize = image.shape[:2]
            # convert percentage into pixel amount
            if self.padding[2] <= 1 and isinstance(self.padding[2], float):
                self.padding[2] = int(self.padding[2] * ysize)
            # different padding shape for grayscale and colored image
            if len(image.shape) > 2:
                padding_shape = (self.padding[2], xsize, image.shape[2])
            else:
                padding_shape = (self.padding[2], xsize)
            # create the padding image
            if self.padding_type == "duplicate":
                image_padding = image[-self.padding[2] :, :].copy()
                if mask is not None:
                    mask_padding = mask[-self.padding[2] :, :].copy()
            elif self.padding_type == "mirror":
                image_padding = np.flipud(image[: self.padding[2], :].copy())
                if mask is not None:
                    mask_padding = np.flipud(mask[: self.padding[2], :].copy())
            else:
                image_padding = np.full(padding_shape, fill_value=padding_value, dtype="uint8")
                if mask is not None:
                    mask_padding = np.full(padding_shape[:2], fill_value=0, dtype="uint8")
            # combine padding image and original image
            image = np.concatenate([image_padding, image], axis=0)
            if mask is not None:
                mask = np.concatenate([mask_padding, mask], axis=0)

        # padding on bottom side
        if self.padding[3] > 0:
            # get image size
            ysize, xsize = image.shape[:2]
            # convert percentage into pixel amount
            if self.padding[3] <= 1 and isinstance(self.padding[3], float):
                self.padding[3] = int(self.padding[3] * ysize)
            # different padding shape for grayscale and colored image
            if len(image.shape) > 2:
                padding_shape = (self.padding[3], xsize, image.shape[2])
            else:
                padding_shape = (self.padding[3], xsize)
            # create the padding image
            if self.padding_type == "duplicate":
                image_padding = image[: self.padding[3], :].copy()
                if mask is not None:
                    mask_padding = mask[: self.padding[3], :].copy()
            elif self.padding_type == "mirror":
                image_padding = np.flipud(image[-self.padding[3] :, :].copy())
                if mask is not None:
                    mask_padding = np.flipud(mask[-self.padding[3] :, :].copy())
            else:
                image_padding = np.full(padding_shape, fill_value=padding_value, dtype="uint8")
                if mask is not None:
                    mask_padding = np.full(padding_shape[:2], fill_value=0, dtype="uint8")
            # combine padding image and original image
            image = np.concatenate([image, image_padding], axis=0)
            if mask is not None:
                mask = np.concatenate([mask, mask_padding], axis=0)

        # update points location after the padding (we need to add x and y if there's padding on top and left)
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [xpoint + self.padding[0], ypoint + self.padding[2]]

        # # update bounding boxes location after the padding (we need to add x and y if there's padding on top and left)
        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    xspoint + self.padding[0],
                    yspoint + self.padding[2],
                    xepoint + self.padding[0],
                    yepoint + self.padding[2],
                ]

        return image, mask

    def run_scale(self, image, mask, keypoints, bounding_boxes):
        """Scale image size based on the input scaling ratio.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # resize based on scale
        # remove negative value (if any)
        self.scale = list(self.scale)
        self.scale[0] = abs(self.scale[0])
        self.scale[1] = abs(self.scale[1])
        if self.scale[1] != 1 and self.scale[0] != 1:
            scale = random.uniform(self.scale[0], self.scale[1])
            if scale != 1:
                # scale image
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                new_size = (new_width, new_height)
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

                # scale mask and update mask labels after the resize process
                if mask is not None:
                    mask_labels = np.unique(mask).tolist() + [0]
                    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)
                    update_mask_labels(mask, mask_labels)

                # scale keypoints
                if keypoints is not None:
                    for name, points in keypoints.items():
                        for i, (xpoint, ypoint) in enumerate(points):
                            points[i] = [round(xpoint * scale), round(ypoint * scale)]

                # scale bounding boxes
                if bounding_boxes is not None:
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        bounding_boxes[i] = [
                            round(xspoint * scale),
                            round(yspoint * scale),
                            round(xepoint * scale),
                            round(yepoint * scale),
                        ]

        return image, mask

    def run_translation(self, image, mask, keypoints, bounding_boxes):
        """Translate image based on the input translation value.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """
        ysize, xsize = image.shape[:2]
        if self.translation[0] <= 1 and self.translation[0] >= -1 and isinstance(self.translation[0], float):
            self.translation = list(self.translation)
            self.translation[0] = int(self.translation[0] * xsize)
        if self.translation[1] <= 1 and self.translation[1] >= -1 and isinstance(self.translation[1], float):
            self.translation = list(self.translation)
            self.translation[1] = int(self.translation[1] * ysize)

        image_new = np.full_like(image, fill_value=255, dtype="uint8")
        if mask is not None:
            mask_new = np.full((image.shape[0], image.shape[1]), fill_value=0, dtype="uint8")
        offset_x = self.translation[0]
        offset_y = self.translation[1]

        # x translation
        if offset_x > 0:
            image_new[:, offset_x:] = image[:, :-offset_x]
            image = image_new
            if mask is not None:
                mask_new[:, offset_x:] = mask[:, :-offset_x]
                mask = mask_new
        elif offset_x < 0:
            image_new[:, :offset_x] = image[:, abs(offset_x) :]
            image = image_new
            if mask is not None:
                mask_new[:, :offset_x] = mask[:, abs(offset_x) :]
                mask = mask_new

        image_new = np.full_like(image, fill_value=255, dtype="uint8")
        if mask is not None:
            mask_new = np.full((image.shape[0], image.shape[1]), fill_value=0, dtype="uint8")

        # y translation
        if offset_y > 0:
            image_new[offset_y:, :] = image[:-offset_y, :]
            image = image_new
            if mask is not None:
                mask_new[offset_y:, :] = mask[:-offset_y, :]
                mask = mask_new
        elif offset_y < 0:
            image_new[:offset_y, :] = image[abs(offset_y) :, :]
            image = image_new
            if mask is not None:
                mask_new[:offset_y, :] = mask[abs(offset_y) :, :]
                mask = mask_new

        # translate keypoints
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [xpoint + offset_x, ypoint + offset_y]

        # translate bounding boxes
        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    xspoint + offset_x,
                    yspoint + offset_y,
                    xepoint + offset_x,
                    yepoint + offset_y,
                ]

        return image, mask

    def run_flip(self, image, mask, keypoints, bounding_boxes):
        """Flip image left-right or up-down based on the input flipping flags.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # flip left right
        if self.fliplr:
            ysize, xsize = image.shape[:2]
            # flip left right on image
            image = np.fliplr(image)
            # flip left right on mask
            if mask is not None:
                mask = np.fliplr(mask)
            # flip left right on keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xsize - 1 - xpoint, ypoint]
            # flip left right on bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    bounding_boxes[i] = [
                        xsize - 1 - xspoint,
                        yspoint,
                        xsize - 1 - xepoint,
                        yepoint,
                    ]

        # flip up down
        if self.flipud:
            ysize, xsize = image.shape[:2]
            # flip up down on image
            image = np.flipud(image)
            # flip up down on mask
            if mask is not None:
                mask = np.flipud(mask)
            # flip up down on keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xpoint, ysize - 1 - ypoint]
            # flip up down on bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    bounding_boxes[i] = [
                        xspoint,
                        ysize - 1 - yspoint,
                        xepoint,
                        ysize - 1 - yepoint,
                    ]

        return image, mask

    def run_rotation(self, image, mask, keypoints, bounding_boxes):
        """Rotate image based on the input rotation angle.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        """

        # generate random angle
        if (self.rotate_range[0] != 0) | (self.rotate_range[1] != 0):
            angle = random.randint(self.rotate_range[0], self.rotate_range[1])
        else:
            angle = 0
        # rotate image
        if angle != 0:
            ysize, xsize = image.shape[:2]

            # rotate image
            image = rotate_image_PIL(image, angle, expand=1)

            # rotate mask
            if mask is not None:
                mask_labels = np.unique(mask).tolist() + [0]
                mask = rotate_image_PIL(mask, angle, expand=1)
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
                rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, -angle)

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
                rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, -angle)

        return image, mask

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # check and randmize geometric transformations
            if self.randomize:
                self.randomize_parameters(image)

            # crop image
            if self.crop:
                image, mask = self.run_crop(image, mask, keypoints, bounding_boxes)

            # apply padding
            if any(self.padding):
                image, mask = self.run_padding(image, mask, keypoints, bounding_boxes)

            # apply scaling
            image, mask = self.run_scale(image, mask, keypoints, bounding_boxes)

            # translate image based on translation value
            if self.translation[0] != 0 or self.translation[1] != 0:
                image, mask = self.run_translation(image, mask, keypoints, bounding_boxes)

            # apply flipping
            image, mask = self.run_flip(image, mask, keypoints, bounding_boxes)

            # apply rotation
            image, mask = self.run_rotation(image, mask, keypoints, bounding_boxes)

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image] + outputs_extra
            else:
                return image

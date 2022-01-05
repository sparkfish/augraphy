import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class Geometric(Augmentation):
    """Applies basic geometric transformations such as resizing, flips and rotation.

    :param scale: Pair of floats determining new scale of image.
    :type scale: tuple, optional
    :param fliplr: Flag to flip image in left right direction.
    :type fliplr: int, optional
    :param flipud: Flag to flip image in up down direction.
    :type flipud: int, optional
    :param rotate_range: Pair of ints determining the range from which to sample
           the image rotation.
    :type rotate_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scale=(1, 1),
        fliplr=0,
        flipud=0,
        rotate_range=(-5, 5),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.scale = scale
        self.fliplr = fliplr
        self.flipud = flipud
        self.rotate_range = rotate_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Geometry(scale={self.scale}, fliplr={self.fliplr}, flipud={self.flipud}, rotate_range={self.rotate_range}, p={self.p})"

    def rotate_image(self, mat, angle):
        """Rotates an image (angle in degrees) and expands image to avoid
        cropping.
        """
        mat = cv2.bitwise_not(mat)
        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2,
        )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        rotated_mat = cv2.bitwise_not(rotated_mat)
        return rotated_mat

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # resize based on scale
            if self.scale[1] > self.scale[0]:
                scale = random.randint(self.scale[0] * 10, self.scale[1] * 10) / 10
                if scale > 0:
                    new_width = int(image.shape[1] * scale)
                    new_height = int(image.shape[0] * scale)
                    new_size = (new_width, new_height)
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            # flip left right
            if self.fliplr:
                image = np.fliplr(image)

            # flip up down
            if self.flipud:
                image = np.flipud(image)

            # generate random angle
            if (self.rotate_range[0] != 0) | (self.rotate_range[1] != 0):
                angle = random.randint(self.rotate_range[0], self.rotate_range[1])
            else:
                angle = 0
            # rotate image
            if angle != 0:
                image = self.rotate_image(image, angle)

            return image

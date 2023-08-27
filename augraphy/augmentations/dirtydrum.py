import random

import cv2
import numpy as np
from PIL import Image
from sklearn.datasets import make_blobs

from augraphy.base.augmentation import Augmentation
from augraphy.utilities import *


class DirtyDrum(Augmentation):
    """Emulates dirty drum effect by creating stripes of vertical and
    horizontal noises.

    :param line_width_range: Pair of ints determining the range from which the
           width of a dirty drum line is sampled.
    :type line_width_range: tuple, optional
    :param line_concentration: Concentration or number of dirty drum lines.
    :type line_concentration: float, optional
    :param direction: Direction of effect, -1=random, 0=horizontal, 1=vertical, 2=both.
    :type direction: int, optional
    :param noise_intensity: Intensity of dirty drum effect, recommended value
           range from 0.8 to 1.0.
    :type noise_intensity: float, optional
    :param noise_value: Tuple of ints to determine value of dirty drum noise.
    :type noise_value: tuple, optional
    :param ksize: Tuple of height/width pairs from which to sample the kernel
           size. Higher value increases the spreadness of stripes.
    :type ksizes: tuple, optional
    :param sigmaX: Standard deviation of the kernel along the x-axis.
    :type sigmaX: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        line_width_range=(1, 4),
        line_concentration=0.1,
        direction=-1,
        noise_intensity=0.5,
        noise_value=(0, 30),
        ksize=(3, 3),
        sigmaX=0,
        p=1,
    ):
        super().__init__(p=p)
        self.line_width_range = line_width_range
        self.line_concentration = line_concentration
        self.direction = direction
        self.noise_intensity = noise_intensity
        self.noise_value = noise_value
        self.ksize = ksize
        self.sigmaX = sigmaX

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DirtyDrum(line_width_range={self.line_width_range}, line_concentration={self.line_concentration}, direction={self.direction}, noise_intensity={self.noise_intensity}, noise_value={self.noise_value}, ksize={self.ksize}, sigmaX={self.sigmaX},p={self.p})"

    def blend(self, img, img_dirty):
        """Blend two images to produce DirtyDrum effect。

        :param img: The background image to apply the blending function.
        :type img: numpy.array (numpy.uint8)
        :param img_dirty: The foreground image to apply the blending function.
        :type img_dirty: numpy.array (numpy.uint8)
        """

        ob = OverlayBuilder(
            "darken",
            img_dirty.astype("uint8"),
            img,
            1,
            (1, 1),
            "center",
            0,
        )
        return ob.build_overlay()

    def add_noise(self, img, y0, yn, x0, xn):
        """Add noise to stripe of image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param y0: The y start coordinate of the image stripe.
        :type y0: int
        :param yn: The y end coordinate of the image stripe.
        :type yn: int
        :param x0: The x start coordinate of the image stripe.
        :type x0: int
        :param xn: The x end coordinate of the image stripe.
        :type xn: int
        """
        ysize, xsize = img.shape[:2]

        # generate parameter values for noise generation
        # get x and y difference
        x_dif = int(xn - x0)
        y_dif = int(yn - y0)

        # generate deviation value
        random_deviation = max(1, int(min(x_dif, y_dif) / 10))

        # generate min and max of noise clusters
        n_cluster_min = max(
            int(x_dif * y_dif * (self.noise_intensity / 150)) - random_deviation,
            1,
        )
        n_cluster_max = max(
            int(x_dif * y_dif * (self.noise_intensity / 150)) + random_deviation,
            1,
        )
        # generate min and max of noise samples
        n_samples_min = max(
            int(x_dif * y_dif * (self.noise_intensity / 70)) - random_deviation,
            1,
        )
        n_samples_max = max(
            int(x_dif * y_dif * (self.noise_intensity / 70)) + random_deviation,
            1,
        )
        # generate min and max fr std range
        std_min = max(int(x_dif / 2) - random_deviation, 1)
        std_max = max(int(x_dif / 2) + random_deviation, 1)

        # generate randomized cluster of samples
        n_samples = [
            random.randint(n_samples_min, n_samples_max) for _ in range(random.randint(n_cluster_min, n_cluster_max))
        ]

        # get randomized std
        std = random.randint(std_min, std_max)

        # x center of noise
        center_x = x0 + int((xn - x0) / 2)

        # generate clusters of noises
        generated_points_x, point_group = make_blobs(
            n_samples=n_samples,
            center_box=(center_x, center_x),
            cluster_std=std,
            n_features=1,
        )

        # generate clusters of noises
        generated_points_y, point_group = make_blobs(
            n_samples=n_samples,
            center_box=(y0, yn),
            cluster_std=std,
            n_features=1,
        )

        # generate x and y points of noise
        generated_points_x = generated_points_x.astype("int")
        generated_points_y = generated_points_y.astype("int")

        # remove invalid points
        ind_delete_x1 = np.where(generated_points_x < 0)
        ind_delete_x2 = np.where(generated_points_x >= xn * 3)
        ind_delete_x3 = np.where(generated_points_x >= xsize)
        ind_delete_y1 = np.where(generated_points_y < 0)
        ind_delete_y2 = np.where(generated_points_y >= yn)

        ind_delete = np.concatenate(
            (ind_delete_x1, ind_delete_x2, ind_delete_x3, ind_delete_y1, ind_delete_y2),
            axis=1,
        )
        generated_points_x = np.delete(generated_points_x, ind_delete, axis=0)
        generated_points_y = np.delete(generated_points_y, ind_delete, axis=0)

        # generate noise
        img[generated_points_y, generated_points_x] = random.randint(
            self.noise_value[0],
            self.noise_value[1],
        )

    def create_dirty_mask(self, img, line_width_range=(6, 18), axis=1):
        """Create mask for drity drum effect。

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param line_width_range: Pair of ints determining the range from which the width of a dirty drum line is sampled.
        :type line_width_range: tuple
        :param axis: The direction of noise line, 0 - horizontal, 1 - vertical.
        :type axis: int
        """

        # initialization
        img_dirty = np.ones_like(img).astype("uint8") * 255
        ysize, xsize = img.shape[:2]

        x = 0
        # generate initial random strip width
        current_width = random.randint(
            line_width_range[0],
            line_width_range[1],
        ) * random.randint(1, 5)

        # flag to break
        f_break = 0

        while True:
            # create random space between lines
            if random.random() > 1 - self.line_concentration:
                # coordinates of stripe
                ys = 0
                ye = ysize
                xs = x
                xe = x + (current_width * 2)

                # apply noise to last patch
                self.add_noise(img_dirty, ys, ye, xs, xe)

            # increment on next x start location
            x += current_width * random.randint(1, 3)

            # generate next random strip width
            current_width = random.randint(
                line_width_range[0],
                line_width_range[1],
            ) * random.randint(1, 5)

            # if next strip > image width, set it to fit into image width
            if x + (current_width) > xsize - 1:
                current_width = int((xsize - 1 - x) / 2)
                if f_break:
                    break
                else:
                    f_break = 1

        # for horizontal stripes, rotate current image
        if axis == 0:
            img_dirty = np.rot90(img_dirty, random.choice((1, 3)))
            # resize after rotation
            img_dirty = cv2.resize(img_dirty, (img.shape[1], img.shape[0]))

        return img_dirty

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # check and convert image into BGR format
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if self.direction == -1:
                # Select random direction
                direction = random.choice([0, 1, 2])
            else:
                direction = self.direction

            if direction == 0:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 0)
                # Apply gaussian blur to mask of dirty drum
                image_dirty = cv2.GaussianBlur(
                    image_dirty,
                    ksize=self.ksize,
                    sigmaX=self.sigmaX,
                )
            elif direction == 1:
                # Create directional masks for dirty drum effect
                image_dirty = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                image_dirty = cv2.GaussianBlur(
                    image_dirty,
                    ksize=self.ksize,
                    sigmaX=self.sigmaX,
                )
            else:
                # Create directional masks for dirty drum effect
                image_dirty_h = self.create_dirty_mask(image, self.line_width_range, 0)
                image_dirty_v = self.create_dirty_mask(image, self.line_width_range, 1)
                # Apply gaussian blur to mask of dirty drum
                image_dirty_h = cv2.GaussianBlur(
                    image_dirty_h,
                    ksize=self.ksize,
                    sigmaX=self.sigmaX,
                )
                image_dirty_v = cv2.GaussianBlur(
                    image_dirty_v,
                    ksize=self.ksize,
                    sigmaX=self.sigmaX,
                )
                # Blend image with the masks of dirty drum effect
                image_dirty = self.blend(image_dirty_v, image_dirty_h)

            image_dirty_drum = self.blend(image, image_dirty)

            # return image follows the input image color channel
            if is_gray:
                image_dirty_drum = cv2.cvtColor(image_dirty_drum, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_dirty_drum = np.dstack((image_dirty_drum, image_alpha))

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_dirty_drum] + outputs_extra
            else:
                return image_dirty_drum

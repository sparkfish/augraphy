import random

import cv2
import numpy as np

from augraphy.augmentations.noisylines import NoisyLines
from augraphy.base.augmentation import Augmentation


class Squish(Augmentation):
    """Creates a squish effect by removing a fixed horizontal or vertical section of the image.

    :param squish_direction: Direction of the squish effect.
        Use 0 for horizontal squish, 1 for vertical squish, 2 for both directions.
        Use "random" to generate random direction.
    :type squish_direction: int or string, optional
    :param squish_location: List of ints determining the location of squish effect.
        If direction of squish effect is horizontal, the value determines the row coordinate of the lines.
        If direction of squish effect is vertical, the value determines the column coordinate of the lines.
        If both directions are selected, the value determines both row and column coordinate of the lines.
    :type squish_location: list, optional
    :param squish_number_range: Tuple of ints determining the number of squish effect.
    :type squish_number_range: tuple, optional
    :param squish_distance_range: Tuple of ints determining the distance of squish effect.
    :type squish_distance_range: tuple, optional
    :param squish_line: Flag to enable drawing of line in each squish effect.
    :type squish_line: int, optional
    :param squish_line_thickness_range: Tuple of ints determing the thickness of squish line.
    :type squish_line_thickness_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        squish_direction="random",
        squish_location="random",
        squish_number_range=(5, 10),
        squish_distance_range=(5, 7),
        squish_line="random",
        squish_line_thickness_range=(1, 1),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.squish_direction = squish_direction
        self.squish_location = squish_location
        self.squish_number_range = squish_number_range
        self.squish_distance_range = squish_distance_range
        self.squish_line = squish_line
        self.squish_line_thickness_range = squish_line_thickness_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Squish(squish_direction={self.squish_direction}, squish_location={self.squish_location}, squish_number_range={self.squish_number_range}, squish_distance_range={self.squish_distance_range}, squish_line={self.squish_line}, squish_line_thickness_range={self.squish_line_thickness_range}, p={self.p})"

    def apply_squish(self, image):
        """Core function to apply the squish effect.

        :param image: The input image.
        :type image: numpy array
        """

        ysize, xsize = image.shape[:2]

        # generate random squish number
        squish_number = random.randint(self.squish_number_range[0], self.squish_number_range[1])

        # generate
        if self.squish_location == "random":
            squish_ys = random.sample(range(0, ysize - 1), squish_number)
        else:
            squish_ys = self.squish_location

        # reverse sort to squish from bottom so that squish location won't be affected after multiple squish iterations
        squish_ys.sort(reverse=True)
        squish_distance_total = 0
        squish_distances = []

        for y in squish_ys:
            # apply squish effect based on the distance
            squish_distance = random.randint(self.squish_distance_range[0], self.squish_distance_range[1])
            image[y:-squish_distance, :] = image[y + squish_distance :, :]
            squish_distances.append(squish_distance)
            # add total squish distance so that we can remove it later
            squish_distance_total += squish_distance
        image = image[:-squish_distance_total, :]

        # generate flag for squish line
        if self.squish_line == "random":
            squish_line = 1
        else:
            squish_line = self.squish_line
        # generate lines
        if squish_line:
            squish_lines_y = []
            # reduce y location when there's multiple squishes
            for i, squish_y in enumerate(squish_ys, start=1):
                squish_line_y = squish_y - sum(squish_distances[i:])
                if self.squish_line == "random":
                    if random.choice([0, 1]) > 0:
                        squish_lines_y.append(squish_line_y)
                else:
                    squish_lines_y.append(squish_line_y)
            noisy_lines = NoisyLines(
                noisy_lines_direction=0,
                noisy_lines_location=squish_lines_y,
                noisy_lines_number_range=(1, 1),
                noisy_lines_color=(0, 0, 0),
                noisy_lines_thickness_range=self.squish_line_thickness_range,
                noisy_lines_random_noise_intensity_range=(0.01, 0.1),
                noisy_lines_length_interval_range=(0, 0),
                noisy_lines_gaussian_kernel_value_range=(1, 1),
                noisy_lines_overlay_method="ink_to_paper",
            )
            image = noisy_lines(image)

        return image

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # generate squish direction
            if self.squish_direction == "random":
                squish_direction = random.choice([0, 1, 2])
            else:
                squish_direction = self.squish_direction

            # horizontal squish
            if squish_direction == 0:
                image_output = self.apply_squish(image)
            # vertical squish
            elif squish_direction == 1:
                image_output = np.rot90(self.apply_squish(np.rot90(image, 3)), 1)
            # horizontal and vertical squish
            else:
                image_output = self.apply_squish(image)
                image_output = np.rot90(self.apply_squish(np.rot90(image_output, 3)), 1)

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

            return image_output

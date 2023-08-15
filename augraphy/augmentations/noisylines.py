import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class NoisyLines(Augmentation):
    """Create noisy lines by drawing horizontal or vertical lines in a fixed intervals.

    :param noisy_lines_direction: Direction of the lines.
        Use 0 for horizontal lines, 1 for vertical lines, 2 for both directions.
        Use "random" to generate random direction.
    :type noisy_lines_direction: int or string, optional
    :param noisy_lines_location: List of ints determining the location of lines.
        If direction of lines is horizontal, the value determines the row coordinate of the lines.
        If direction of lines is vertical, the value determines the column coordinate of the lines.
        If both directions are selected, the value determines both rw and column coordinate of the lines.
    :type noisy_lines_location: list, optional
    :param noisy_lines_number_range: Tuple of ints determining the number of lines.
    :type noisy_lines_number_range: tuple, optional
    :param noisy_lines_color: The color of the lines in BGR.
    :type noisy_lines_color: tuple, optional
    :param noisy_lines_thickness_range: Tuple of ints determining the thickness of the lines.
    :type noisy_lines_thickness_range: tuple, optional
    :param noisy_lines_random_noise_intensity_range: Tuple of floats determining the random noise of the lines.
    :type noisy_lines_random_noise_intensity_range: tuple, optional
    :param noisy_lines_length_interval_range: Tuple of ints determining the space interval of each line.
    :type noisy_lines_length_interval_range: tuple, optional
    :param noisy_lines_gaussian_kernel_value_range: Tuple of ints determining the Gaussian kernel value.
    :type noisy_lines_gaussian_kernel_value_range: tuple, optional
    :param noisy_lines_overlay_method: The method to overlay mask of lines into the input image.
    :type noisy_lines_overlay_method: string, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        noisy_lines_direction="random",
        noisy_lines_location="random",
        noisy_lines_number_range=(5, 20),
        noisy_lines_color=(0, 0, 0),
        noisy_lines_thickness_range=(1, 2),
        noisy_lines_random_noise_intensity_range=(0.01, 0.1),
        noisy_lines_length_interval_range=(0, 100),
        noisy_lines_gaussian_kernel_value_range=(3, 5),
        noisy_lines_overlay_method="ink_to_paper",
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.noisy_lines_direction = noisy_lines_direction
        self.noisy_lines_location = noisy_lines_location
        self.noisy_lines_number_range = noisy_lines_number_range
        self.noisy_lines_color = noisy_lines_color
        self.noisy_lines_thickness_range = noisy_lines_thickness_range
        self.noisy_lines_random_noise_intensity_range = noisy_lines_random_noise_intensity_range
        self.noisy_lines_length_interval_range = noisy_lines_length_interval_range
        self.noisy_lines_gaussian_kernel_value_range = noisy_lines_gaussian_kernel_value_range
        self.noisy_lines_overlay_method = noisy_lines_overlay_method

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"NoisyLines(noisy_lines_direction={self.noisy_lines_direction}, noisy_lines_location={self.noisy_lines_location}, noisy_lines_number_range={self.noisy_lines_number_range}, noisy_lines_color={self.noisy_lines_color}, noisy_lines_thickness_range={self.noisy_lines_thickness_range}, noisy_lines_random_noise_intensity_range={self.noisy_lines_random_noise_intensity_range}, noisy_lines_length_interval_range={self.noisy_lines_length_interval_range}, noisy_lines_gaussian_kernel_value_range={self.noisy_lines_gaussian_kernel_value_range}, noisy_lines_overlay_method={self.noisy_lines_overlay_method}, p={self.p})"

    def draw_noisy_lines(self, image):
        """Core function to draw noisy lines in the input image

        :param image: The input image.
        :type image: numpy array
        """

        ysize, xsize = image.shape[:2]

        # mask of lines
        image_mask = np.full_like(image, fill_value=255, dtype="uint8")

        # generate random number of lines
        noisy_lines_number = random.randint(self.noisy_lines_number_range[0], self.noisy_lines_number_range[1])

        # draw lines
        if self.noisy_lines_location == "random":
            y_coordinates = random.sample(range(0, ysize - 1), noisy_lines_number)
        else:
            y_coordinates = self.noisy_lines_location

        y_coordinates.sort()

        for y_coordinate in y_coordinates:

            noisy_lines_thickness = random.randint(
                self.noisy_lines_thickness_range[0],
                self.noisy_lines_thickness_range[1],
            )
            cv2.line(
                image_mask,
                (0, y_coordinate),
                (xsize - 1, y_coordinate),
                self.noisy_lines_color,
                thickness=noisy_lines_thickness,
            )

            noisy_lines_length_interval = random.randint(
                self.noisy_lines_length_interval_range[0],
                self.noisy_lines_length_interval_range[1],
            )
            # remove some section of lines based on interval
            start_x = random.randint(0, noisy_lines_length_interval)
            if noisy_lines_thickness > 1 and noisy_lines_length_interval > 0:
                half_thickness = int(np.ceil(noisy_lines_thickness / 2))
                for new_y_coordinate in range(y_coordinate - half_thickness, y_coordinate + half_thickness + 1):
                    if new_y_coordinate >= 0 and new_y_coordinate < ysize:

                        for x in range(start_x, xsize, noisy_lines_length_interval * 2):
                            image_mask[
                                new_y_coordinate,
                                x + noisy_lines_length_interval : x + (noisy_lines_length_interval * 2),
                            ] = 255

        # apply noise in the line
        image_random = np.random.uniform(0, 1, size=image.shape[:2])
        noisy_lines_random_noise_intensity = random.uniform(
            self.noisy_lines_random_noise_intensity_range[0],
            self.noisy_lines_random_noise_intensity_range[1],
        )
        image_mask[image_random < noisy_lines_random_noise_intensity] = 255

        # apply gaussian blur
        gaussian_kernel_value = random.randint(
            self.noisy_lines_gaussian_kernel_value_range[0],
            self.noisy_lines_gaussian_kernel_value_range[1],
        )
        # gaussian kernel must be odd
        if not gaussian_kernel_value % 2:
            gaussian_kernel_value += 1
        image_mask = cv2.GaussianBlur(image_mask, (gaussian_kernel_value, gaussian_kernel_value), cv2.BORDER_DEFAULT)

        ob = OverlayBuilder(
            self.noisy_lines_overlay_method,
            image_mask,
            image,
            1,
            (1, 1),
            "center",
            0,
        )

        image_output = ob.build_overlay()

        return image_output

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # generate lines direction
            if self.noisy_lines_direction == "random":
                noisy_lines_direction = random.choice([0, 1, 2])
            else:
                noisy_lines_direction = self.noisy_lines_direction

            # horizontal lines
            if noisy_lines_direction == 0:
                image_output = self.draw_noisy_lines(image)
            # vertical lines
            elif noisy_lines_direction == 1:
                # use copy to solve the problem mentioned here:
                # https://stackoverflow.com/questions/16461560/layout-of-the-output-array-img-is-incompatible-with-cvmat-stepndims-1-el
                image_rotated = np.rot90(image, 3).copy()
                image_output = np.rot90(self.draw_noisy_lines(image_rotated), 1).copy()
            # horizontal and vertical lines
            else:
                image_output = np.rot90(self.draw_noisy_lines(image), 3).copy()
                image_output = np.rot90(self.draw_noisy_lines(image_output), 1).copy()

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

            return image_output

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

    def apply_squish(self, image, mask, keypoints, bounding_boxes, squish_direction):
        """Core function to apply the squish effect.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8)
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list
        :param squish_direction: Direction of squish effect, where 0 = horizontal, 1 = vertical.
        :type squish_direction: int
        """

        ysize, xsize = image.shape[:2]

        # generate random squish number
        squish_number = random.randint(self.squish_number_range[0], self.squish_number_range[1])

        # generate
        if self.squish_location == "random":
            # vertical
            if squish_direction:
                squish_coordinates = random.sample(range(0, xsize - 1), squish_number)
            # horizontal
            else:
                squish_coordinates = random.sample(range(0, ysize - 1), squish_number)
        else:
            squish_coordinates = self.squish_location

        # reverse sort to squish from last element so that squish location won't be affected after multiple squish iterations
        squish_coordinates.sort(reverse=True)
        squish_distance_total = 0
        squish_distances = []

        for coordinate in squish_coordinates:
            # apply squish effect based on the distance
            squish_distance = random.randint(self.squish_distance_range[0], self.squish_distance_range[1])
            # vertical
            if squish_direction:
                # image
                image[:, coordinate:-squish_distance] = image[:, coordinate + squish_distance :]
                # mask
                if mask is not None:
                    mask[:, coordinate:-squish_distance] = mask[:, coordinate + squish_distance :]
                # keypoints
                if keypoints is not None:
                    for name, points in keypoints.items():
                        remove_indices = []
                        for i, (xpoint, ypoint) in enumerate(points):
                            # remove keypoints in squish box
                            if xpoint >= coordinate and xpoint < coordinate + squish_distance:
                                remove_indices.append(i)
                            # reduce coordinate value if points > coordinate + squish_distance
                            elif xpoint >= coordinate + squish_distance:
                                xpoint -= squish_distance
                                points[i] = [xpoint, ypoint]
                        # remove points
                        while remove_indices:
                            points.pop(remove_indices.pop())
                # bounding boxes
                if bounding_boxes is not None:
                    remove_indices = []
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        # both x points are inside squish coordinate
                        if (
                            xspoint >= coordinate
                            and xspoint < coordinate + squish_distance
                            and xepoint >= coordinate
                            and xepoint < coordinate + squish_distance
                        ):
                            remove_indices.append(i)
                        # start point is in the squish box
                        elif xspoint >= coordinate and xspoint < coordinate + squish_distance:
                            xspoint = coordinate
                        # end point is in the squish box
                        elif xepoint >= coordinate and xepoint < coordinate + squish_distance:
                            xepoint = coordinate
                        # reduce value by squish distance
                        if xspoint >= coordinate + squish_distance:
                            xspoint -= squish_distance
                        if xepoint >= coordinate + squish_distance:
                            xepoint -= squish_distance
                        bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                    # remove boxes
                    while remove_indices:
                        bounding_boxes.pop(remove_indices.pop())

            # horizontal
            else:
                # image
                image[coordinate:-squish_distance, :] = image[coordinate + squish_distance :, :]
                # mask
                if mask is not None:
                    mask[coordinate:-squish_distance, :] = mask[coordinate + squish_distance :, :]
                # keypoints
                if keypoints is not None:
                    for name, points in keypoints.items():
                        remove_indices = []
                        for i, (xpoint, ypoint) in enumerate(points):
                            # remove keypoints in squish box
                            if ypoint >= coordinate and ypoint < coordinate + squish_distance:
                                remove_indices.append(i)
                            # reduce coordinate value if points > coordinate + squish_distance
                            elif ypoint >= coordinate + squish_distance:
                                ypoint -= squish_distance
                                points[i] = [xpoint, ypoint]
                        # remove points
                        while remove_indices:
                            points.pop(remove_indices.pop())
                # bounding boxes
                if bounding_boxes is not None:
                    remove_indices = []
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        # both x points are inside squish coordinate
                        if (
                            yspoint >= coordinate
                            and yspoint < coordinate + squish_distance
                            and yepoint >= coordinate
                            and yepoint < coordinate + squish_distance
                        ):
                            remove_indices.append(i)
                        # start point is in the squish box
                        elif yspoint >= coordinate and yspoint < coordinate + squish_distance:
                            yspoint = coordinate
                        # end point is in the squish box
                        elif yepoint >= coordinate and yepoint < coordinate + squish_distance:
                            yepoint = coordinate
                        # reduce value by squish distance
                        if yspoint >= coordinate + squish_distance:
                            yspoint -= squish_distance
                        if yepoint >= coordinate + squish_distance:
                            yepoint -= squish_distance
                        bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                    # remove boxes
                    while remove_indices:
                        bounding_boxes.pop(remove_indices.pop())

            squish_distances.append(squish_distance)
            # add total squish distance so that we can remove it later
            squish_distance_total += squish_distance
        # vertical
        if squish_direction:
            image = image[:, :-squish_distance_total]
            if mask is not None:
                mask = mask[:, :-squish_distance_total]
        # horizontal
        else:
            image = image[:-squish_distance_total, :]
            if mask is not None:
                mask = mask[:-squish_distance_total, :]

        # generate flag for squish line
        if self.squish_line == "random":
            squish_line = 1
        else:
            squish_line = self.squish_line
        # generate lines
        if squish_line:
            squish_lines_coordinates = []
            # reduce y location when there's multiple squishes
            for i, coordinate in enumerate(squish_coordinates, start=1):
                squish_lines_coordinate = coordinate - sum(squish_distances[i:])
                if self.squish_line == "random":
                    if random.choice([0, 1]) > 0:
                        squish_lines_coordinates.append(squish_lines_coordinate)
                else:
                    squish_lines_coordinates.append(squish_lines_coordinate)
            noisy_lines = NoisyLines(
                noisy_lines_direction=squish_direction,
                noisy_lines_location=squish_lines_coordinates,
                noisy_lines_number_range=(1, 1),
                noisy_lines_color=(0, 0, 0),
                noisy_lines_thickness_range=self.squish_line_thickness_range,
                noisy_lines_random_noise_intensity_range=(0.01, 0.1),
                noisy_lines_length_interval_range=(0, 0),
                noisy_lines_gaussian_kernel_value_range=(1, 1),
                noisy_lines_overlay_method="ink_to_paper",
            )
            image = noisy_lines(image)

        return image, mask

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

            if squish_direction == 2:
                random_direction = random.randint(0, 1)
                image_output, mask = self.apply_squish(
                    image,
                    mask,
                    keypoints,
                    bounding_boxes,
                    squish_direction=random_direction,
                )
                image_output, mask = self.apply_squish(
                    image_output,
                    mask,
                    keypoints,
                    bounding_boxes,
                    squish_direction=1 - random_direction,
                )
            else:
                image_output, mask = self.apply_squish(image, mask, keypoints, bounding_boxes, squish_direction)

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

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

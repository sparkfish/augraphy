import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class LCDScreenPattern(Augmentation):
    """Creates a LCD Screen Pattern effect by overlaying different line patterns into image.

    :param pattern_type: Types of pattern. Use "random" for random pattern.
        Select from "Vertical", "Horizontal", "Forward_Diagonal", "Back_Diagonal" and "Cross".
    :type pattern_type: string, optional
    :param pattern_value_range: Tuple of ints determining the value of the pattern.
    :type pattern_value_range: tuple, optional
    :param pattern_skip_distance_range: Tuples of ints determining the distance between lines in each pattern.
        This is not valid for pattern type of "Cross".
    :type pattern_skip_distance_range: tuple, optional
    :param pattern_overlay_method: The method to overlay pattern into image using OverlayBuilder.
        The default value is "darken".
    :type pattern_overlay_method: string, optional
    :param pattern_overlay_alpha: The alpha value for the overlay method uses alpha value.
    :type pattern_overlay_alpha: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        pattern_type="random",
        pattern_value_range=(0, 16),
        pattern_skip_distance_range=(3, 5),
        pattern_overlay_method="darken",
        pattern_overlay_alpha=0.3,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.pattern_type = pattern_type
        self.pattern_value_range = pattern_value_range
        self.pattern_skip_distance_range = pattern_skip_distance_range
        self.pattern_overlay_method = pattern_overlay_method
        self.pattern_overlay_alpha = pattern_overlay_alpha

    def __repr__(self):
        return f"LCDScreenPattern(pattern_type={self.pattern_type}, pattern_value_range={self.pattern_value_range}, pattern_skip_distance_range={self.pattern_skip_distance_range}, pattern_overlay_method={self.pattern_overlay_method}, pattern_overlay_alpha={self.pattern_overlay_alpha}, p={self.p})"

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

            ysize, xsize = image.shape[:2]

            # get types of pattern
            if self.pattern_type == "random":
                pattern_type = random.choice(["Vertical", "Horizontal", "Forward_Diagonal", "Back_Diagonal", "Cross"])
            else:
                pattern_type = self.pattern_type

            # get value
            value = random.randint(self.pattern_value_range[0], self.pattern_value_range[1])

            # initialize image
            image_pattern = np.full_like(image, fill_value=255, dtype="uint8")

            pattern_skip_distance = random.randint(
                self.pattern_skip_distance_range[0],
                self.pattern_skip_distance_range[1],
            )

            if pattern_type == "Vertical":
                image_pattern[:, ::pattern_skip_distance] = value

            elif pattern_type == "Horizontal":
                image_pattern[::pattern_skip_distance, :] = value

            elif pattern_type == "Forward_Diagonal":
                # minimum skip size
                pattern_skip_distance = max(3, pattern_skip_distance)

                y, x = np.meshgrid(np.arange(ysize), np.arange(xsize), indexing="ij")

                # Create diagonal lines pattern
                image_pattern = ((x + y) % pattern_skip_distance == 0).astype(np.uint8) * 255
                image_pattern = 255 - image_pattern
                image_pattern[image_pattern == 0] = value

                # Convert from gray to BGR
                if len(image.shape) > 2:
                    image_pattern = cv2.cvtColor(image_pattern, cv2.COLOR_GRAY2BGR)

            elif pattern_type == "Back_Diagonal":
                # minimum skip size
                pattern_skip_distance = max(3, pattern_skip_distance)

                y, x = np.meshgrid(np.arange(ysize), np.arange(xsize), indexing="ij")

                # Create diagonal lines pattern
                image_pattern = ((x - y) % pattern_skip_distance == 0).astype(np.uint8) * 255
                image_pattern = 255 - image_pattern
                image_pattern[image_pattern == 0] = value

                # Convert from gray to BGR
                if len(image.shape) > 2:
                    image_pattern = cv2.cvtColor(image_pattern, cv2.COLOR_GRAY2BGR)

            else:
                image_pattern[::2, ::2] = value
                image_pattern[1::2, 1::2] = value

            # blend pattern into image
            ob = OverlayBuilder(
                self.pattern_overlay_method,
                image_pattern,
                image,
                1,
                (1, 1),
                "center",
                0,
                self.pattern_overlay_alpha,
            )
            image_output = ob.build_overlay()

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

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

import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class ReflectedLight(Augmentation):
    """Creates reflected light effect by drawing ellipses of different brightness.

    :param reflected_light_smoothness: Pair of floats determining the smoothness of ellipse. The value should be in range of 0 to 1.
            It runs slower when the smoothness value is higher.
    :type reflected_light_smoothness: float, optional
    :param reflected_light_internal_radius_range: Pair of ints determining the major length of non fading area (centroid) of ellipse.
    :type reflected_light_internal_radius_range: tuple, optional
    :param reflected_light_external_radius_range: Pair of ints determining the major length of fading area of ellipse.
    :type reflected_light_external_radius_range: tuple, optional
    :param reflected_light_minor_major_ratio_range: Pair of floats determining the ratio of minor length to major length of the ellipse.
    :type reflected_light_minor_major_ratio_range: tuple, optional
    :param reflected_light_color: Color of the effect in BGR.
    :type reflected_light_color: tuple, optional
    :param reflected_light_internal_max_brightness_range: Pair of floats determining the max brightness of the internal ellipse.
    :type reflected_light_interal_max_brightness_range: tuple, optional
    :param reflected_light_external_max_brightness_range: Pair of floats determining the max brightness of the external ellipse.
    :type reflected_light_external_max_brightness_range: tuple, optional
    :param reflected_light_location: The location of the effect.
    :type reflected_light_location: tuple, optional
    :param reflected_light_ellipse_angle_range: Pair of ints determining the angle of the ellipse.
    :type reflected_light_ellipse_angle_range: tuple, optional
    :param reflected_light_gaussian_kernel_size_range: Pair of ints determining the Gaussian kernel value.
    :type reflected_light_gaussian_kernel_size_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        reflected_light_smoothness=0.8,
        reflected_light_internal_radius_range=(0.0, 0.2),
        reflected_light_external_radius_range=(0.1, 0.8),
        reflected_light_minor_major_ratio_range=(0.9, 1.0),
        reflected_light_color=(255, 255, 255),
        reflected_light_internal_max_brightness_range=(0.9, 1.0),
        reflected_light_external_max_brightness_range=(0.75, 0.9),
        reflected_light_location="random",
        reflected_light_ellipse_angle_range=(0, 360),
        reflected_light_gaussian_kernel_size_range=(5, 310),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.reflected_light_smoothness = reflected_light_smoothness
        self.reflected_light_internal_radius_range = reflected_light_internal_radius_range
        self.reflected_light_external_radius_range = reflected_light_external_radius_range
        self.reflected_light_minor_major_ratio_range = reflected_light_minor_major_ratio_range
        self.reflected_light_color = reflected_light_color
        self.reflected_light_internal_max_brightness_range = reflected_light_internal_max_brightness_range
        self.reflected_light_external_max_brightness_range = reflected_light_external_max_brightness_range
        self.reflected_light_location = reflected_light_location
        self.reflected_light_ellipse_angle_range = reflected_light_ellipse_angle_range
        self.reflected_light_gaussian_kernel_size_range = reflected_light_gaussian_kernel_size_range

    def __repr__(self):
        return f"ReflectedLight(reflected_light_smoothness={self.reflected_light_smoothness}, reflected_light_internal_radius_range={self.reflected_light_internal_radius_range}, reflected_light_external_radius_range={self.reflected_light_external_radius_range}, reflected_light_minor_major_ratio_range={self.reflected_light_minor_major_ratio_range},  reflected_light_color={self.reflected_light_color}, reflected_light_internal_max_brightness_range={self.reflected_light_internal_max_brightness_range}, reflected_light_external_max_brightness_range={self.reflected_light_external_max_brightness_range}， reflected_light_location={self.reflected_light_location}, reflected_light_ellipse_angle_range={self.reflected_light_ellipse_angle_range}, reflected_light_gaussian_kernel_size_range={self.reflected_light_gaussian_kernel_size_range},p={self.p})"

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            # convert and make sure image is color image
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
            min_size = min(ysize, xsize)

            # reflected_light color
            if self.reflected_light_color == "random":
                reflected_light_color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            else:
                reflected_light_color = self.reflected_light_color
                # min value of color is 1 so that we can differentiate reflected_light color and background
                if reflected_light_color[0] < 1 or reflected_light_color[1] < 1 or reflected_light_color[0] < 1:
                    reflected_light_color = (
                        max(1, reflected_light_color[0]),
                        max(1, reflected_light_color[1]),
                        max(1, reflected_light_color[2]),
                    )

            # set min color value intensity = 64
            single_color_pixel = np.full((1, 1, 3), fill_value=reflected_light_color, dtype="uint8")
            reflected_light_color_hsv = cv2.cvtColor(single_color_pixel, cv2.COLOR_BGR2HSV)
            reflected_light_color_hsv[:, :, 2][0, 0] = max(64, reflected_light_color_hsv[:, :, 2][0, 0])
            single_color_pixel = cv2.cvtColor(reflected_light_color_hsv, cv2.COLOR_HSV2BGR)
            reflected_light_color = [int(color) for color in single_color_pixel[0, 0]]

            # reflected_light internal and external radius
            if self.reflected_light_internal_radius_range[1] <= 1 and isinstance(
                self.reflected_light_internal_radius_range[1],
                float,
            ):
                reflected_light_internal_radius_range = [
                    self.reflected_light_internal_radius_range[0] * min_size,
                    self.reflected_light_internal_radius_range[1] * min_size,
                ]
            else:
                reflected_light_internal_radius_range = self.reflected_light_internal_radius_range
            if self.reflected_light_external_radius_range[1] <= 1 and isinstance(
                self.reflected_light_external_radius_range[1],
                float,
            ):
                reflected_light_external_radius_range = [
                    self.reflected_light_external_radius_range[0] * min_size,
                    self.reflected_light_external_radius_range[1] * min_size,
                ]
            else:
                reflected_light_external_radius_range = self.reflected_light_external_radius_range
            reflected_light_internal_radius = random.randint(
                int(reflected_light_internal_radius_range[0]),
                int(reflected_light_internal_radius_range[1]),
            )
            reflected_light_external_radius = random.randint(
                int(reflected_light_external_radius_range[0]),
                int(reflected_light_external_radius_range[1]),
            )

            # reflected_light centroid location
            if self.reflected_light_location == "random":
                reflected_light_center_x = random.randint(0, xsize - 1)
                reflected_light_center_y = random.randint(0, ysize - 1)
            else:
                # generate reflected_light location  based on image size
                if self.reflected_light_location[0] <= 1 and isinstance(self.reflected_light_location[0], float):
                    reflected_light_center_x = int(self.reflected_light_location[0] * xsize)
                if self.reflected_light_location[1] <= 1 and isinstance(self.reflected_light_location[1], float):
                    reflected_light_center_y = int(self.reflected_light_location[1] * ysize)

            # reflected_light brightness
            reflected_light_internal_max_brightness = random.uniform(
                self.reflected_light_internal_max_brightness_range[0],
                self.reflected_light_internal_max_brightness_range[1],
            )
            reflected_light_external_max_brightness = random.uniform(
                self.reflected_light_external_max_brightness_range[0],
                self.reflected_light_external_max_brightness_range[1],
            )

            # reflected_light ellipse angle
            reflected_light_ellipse_angle = random.randint(
                self.reflected_light_ellipse_angle_range[0],
                self.reflected_light_ellipse_angle_range[1],
            )

            # reflected_light gaussian kernel value
            reflected_light_gaussian_kernel_value = random.randint(
                self.reflected_light_gaussian_kernel_size_range[0],
                self.reflected_light_gaussian_kernel_size_range[1],
            )
            # gaussian kernel value must be odd
            if not reflected_light_gaussian_kernel_value % 2:
                reflected_light_gaussian_kernel_value += 1

            reflected_light_minor_major_ratio = random.uniform(
                self.reflected_light_minor_major_ratio_range[0],
                self.reflected_light_minor_major_ratio_range[1],
            )

            # axes which is the major and minor radius of ellipse
            reflected_light_axes = [
                reflected_light_external_radius + reflected_light_internal_radius,
                int(
                    (reflected_light_external_radius + reflected_light_internal_radius)
                    * reflected_light_minor_major_ratio,
                ),
            ]

            # general background for the reflected_light effect
            image_background = np.zeros_like(image, dtype="uint8")

            # draw each circle of ellipse with increasing alpha, but decreasing axis length, creating an spot of light
            reflected_light_alpha = reflected_light_external_max_brightness - 1

            # compute parameters of each drawing step
            total_diameter = int(reflected_light_external_radius * reflected_light_minor_major_ratio)
            smooth_threshold = 50
            smoothness = max(1, smooth_threshold - (smooth_threshold * self.reflected_light_smoothness))
            total_length = int(total_diameter / smoothness)
            step_length = int(total_diameter / total_length)
            step_alpha = 1 / total_length

            # random image to create random pattern
            image_random = np.random.uniform(0, 1, size=image.shape[:2])
            # external ellipse area
            while True:

                current_reflected_light_alpha = reflected_light_alpha
                # clip value between 0 and 1
                current_reflected_light_alpha = np.clip(current_reflected_light_alpha, 0.0, 1.0)

                # create image and draw circle
                image_background_new = np.zeros_like(image, dtype="uint8")
                # draw ellipse
                cv2.ellipse(
                    image_background_new,
                    (reflected_light_center_x, reflected_light_center_y),
                    reflected_light_axes,
                    reflected_light_ellipse_angle,
                    0,
                    360,
                    reflected_light_color,
                    step_length,
                )

                indices = image_random > min(1.0, current_reflected_light_alpha)
                for i in range(3):
                    image_background_new[:, :, i][indices] = 0

                # draw image with lower alpha into main background
                indices = np.logical_and(image_background_new > 0, image_background == (0, 0, 0))
                image_background[indices] = image_background_new[indices] * current_reflected_light_alpha

                # break when current diameter is the same with internal ellipse diameter
                if total_diameter == reflected_light_internal_radius:
                    break
                # for last loop, fill the gap between external and internal ellipse
                elif total_diameter - step_length < reflected_light_internal_radius:
                    step_length = total_diameter - reflected_light_internal_radius
                    total_diameter -= step_length
                # in each iteration, reduce diameter
                else:
                    total_diameter -= step_length

                # increase alpha of ellipse, but decrease axis length
                reflected_light_alpha += step_alpha
                reflected_light_axes = [reflected_light_axes[0] - step_length, reflected_light_axes[1] - step_length]

            # internal ellipse
            if reflected_light_internal_radius > 0:

                image_background_new = np.zeros_like(image, dtype="uint8")
                # draw the inner ellipse
                cv2.ellipse(
                    image_background_new,
                    (reflected_light_center_x, reflected_light_center_y),
                    reflected_light_axes,
                    reflected_light_ellipse_angle,
                    0,
                    360,
                    reflected_light_color,
                    -1,
                )

                indices = np.logical_and(image_background_new > 0, image_background == (0, 0, 0))
                image_background[indices] = image_background_new[indices] * reflected_light_internal_max_brightness

            # blur ellipse to make it more realisitc
            image_background = cv2.GaussianBlur(
                image_background,
                (reflected_light_gaussian_kernel_value, reflected_light_gaussian_kernel_value),
                0,
            )

            # merge ellipse into image
            image_output = cv2.addWeighted(
                image_background,
                1,
                image,
                1,
                0,
            )

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

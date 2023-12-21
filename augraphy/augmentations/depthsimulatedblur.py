import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class DepthSimulatedBlur(Augmentation):
    """Creates a depth-simulated blur effect from a camera by blurring a small elliptical region of image.

    :param blur_centerr: Center (x,y) of blur effect. Use "random" for random location.
    :type blur_center: tuple or string, optional
    :param blur_major_axes_length_range: Pair of ints determining the value of major axis in the blurring ellipse.
    :type blur_major_axes_length_range: tuple, optional
    :param blur_minor_axes_length_range: Pair of ints determining the value of minor axis in the blurring ellipse.
    :type blur_minor_axes_length_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        blur_center="random",
        blur_major_axes_length_range=(120, 200),
        blur_minor_axes_length_range=(120, 200),
        p=1,
    ):
        super().__init__(p=p)
        self.blur_center = blur_center
        self.blur_major_axes_length_range = blur_major_axes_length_range
        self.blur_minor_axes_length_range = blur_minor_axes_length_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DepthSimulatedBlur(blur_center={self.blur_center}, blur_major_axes_length_range={self.blur_major_axes_length_range}, blur_minor_axes_length_range={self.blur_minor_axes_length_range}, p={self.p})"

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

            ysize, xsize = image.shape[:2]

            # initial gaussian kernel value, will be incremented per iteration
            kernel_value = 3
            gaussian_kernel = [kernel_value, kernel_value]

            axes_major = random.randint(self.blur_major_axes_length_range[0], self.blur_major_axes_length_range[1])
            axes_minor = random.randint(self.blur_minor_axes_length_range[0], self.blur_minor_axes_length_range[1])

            if self.blur_center == "random":
                center_x = random.randint(0, xsize)
                center_y = random.randint(0, ysize)
            else:
                center_x = self.blur_center[0]
                center_y = self.blur_center[0]

            step = 10
            decremental_value = 10
            angle = random.randint(0, 360)  # Angle of rotation (in degrees)
            center_coordinates = (center_x, center_y)
            color = (255, 255, 255)  # BGR color (here, blue)
            thickness = -1  # Line thickness

            image_output = image.copy()

            # it still run slow now, need further optimization
            for i in range(step):

                image_ellipse = np.zeros_like(image, dtype="uint8")

                axes_length = (axes_major, axes_minor)  # Major and minor axes lengths

                # Draw the oval on the image
                cv2.ellipse(image_ellipse, center_coordinates, axes_length, angle, 0, 360, color, thickness)

                # blur image
                image_blur = cv2.GaussianBlur(image, gaussian_kernel, 0)

                # blend blur region into image
                image_output = cv2.seamlessClone(
                    image_output,
                    image_blur,
                    255 - image_ellipse,
                    (int(xsize / 2), int(ysize / 2)),
                    cv2.NORMAL_CLONE,
                )

                # increase gaussian kernel size
                gaussian_kernel[0] += 4
                gaussian_kernel[1] += 4

                # increase major and minor length
                axes_major = max(axes_major - decremental_value, 1)
                axes_minor = max(axes_minor - decremental_value, 1)

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

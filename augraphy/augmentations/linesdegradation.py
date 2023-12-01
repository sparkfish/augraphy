import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class LinesDegradation(Augmentation):
    """Degrades lines by replacing lines formed by image gradients with a different value.

    :param line_roi: Tuple of 4 (x0, y0, xn, yn) to determine the region of interest of the augmentation effect.
             The value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
             x0 (int) = image width  * x0 (float and 0.0 - 1.0);
             y0 (int) = image height * y0 (float and 0.0 - 1.0);
             xn (int) = image width  * xn (float and 0.0 - 1.0);
             yn (int) = image height * yn (float and 0.0 - 1.0)
    :type line_roi: tuple, optional
    :param line_gradient_range: Pair of ints determining range of gradient values (low, high) in detecting the lines.
    :type line_gradient_range: tuple, optional
    :param line_gradient_direction: Set value to 0 for horizontal gradients, 1 for vertical gradients and 2 for both.
    :type line_gradient_direction: tuple, optional
    :param line_split_probability: Pair of floats determining the probability to split long line into shorter lines.
    :type line_split_probability: tuple, optional
    :param line_replacement_value: Pair of ints determining the new value of the detected lines.
    :type line_replacement_value: tuple, optional
    :param line_min_length: Pair of ints determining the minimum length of detected lines.
    :type line_min_length: tuple, optional
    :param line_long_to_short_ratio: Pair of ints determining the threshold ratio of major axis to minor axis of the detected lines.
    :type line_long_to_short_ratio: tuple, optional
    :param ine_replacement_probability: Pair of floats determining the probability to replace the detected lines with new value.
    :type ine_replacement_probability: tuple, optional
    :param line_replacement_thickness: Pair of ints determining the thickness of replaced lines.
    :type line_replacement_thickness: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        line_roi=(0.0, 0.0, 1.0, 1.0),
        line_gradient_range=(32, 255),
        line_gradient_direction=(0, 2),
        line_split_probability=(0.2, 0.4),
        line_replacement_value=(250, 255),
        line_min_length=(30, 40),
        line_long_to_short_ratio=(5, 7),
        line_replacement_probability=(0.4, 0.5),
        line_replacement_thickness=(1, 3),
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.line_roi = line_roi
        self.line_gradient_range = line_gradient_range
        self.line_gradient_direction = line_gradient_direction
        self.line_split_probability = line_split_probability
        self.line_replacement_value = line_replacement_value
        self.line_min_length = line_min_length
        self.line_long_to_short_ratio = line_long_to_short_ratio
        self.line_replacement_probability = line_replacement_probability
        self.line_replacement_thickness = line_replacement_thickness

    def __repr__(self):
        return f"LinesDegradation(line_roi={self.line_roi}, line_gradient_range={self.line_gradient_range}, line_gradient_direction={self.line_gradient_direction}, line_split_probability={self.line_split_probability}, line_replacement_value={self.line_replacement_value}, line_min_length={self.line_min_length}, line_long_to_short_ratio={self.line_long_to_short_ratio}, line_replacement_probability={self.line_replacement_probability}, line_replacement_thickness={self.line_replacement_thickness}, p={self.p})"

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():

            # initialize parameters with random value
            line_split_probability = np.random.uniform(self.line_split_probability[0], self.line_split_probability[1])
            line_min_length = random.randint(self.line_min_length[0], self.line_min_length[1])
            long_to_short_ratio = random.randint(self.line_long_to_short_ratio[0], self.line_long_to_short_ratio[1])
            line_replacement_probability = np.random.uniform(
                self.line_replacement_probability[0],
                self.line_replacement_probability[1],
            )

            # roi
            ysize, xsize = image.shape[:2]
            xstart, ystart, xend, yend = self.line_roi
            # when value is float and in between 0-1, scale it with image size
            if xstart >= 0 and xstart <= 1 and isinstance(xstart, float):
                xstart = int(xstart * xsize)
            if ystart >= 0 and ystart <= 1 and isinstance(ystart, float):
                ystart = int(ystart * ysize)
            if xend >= 0 and xend <= 1 and isinstance(xend, float):
                xend = int(xend * xsize)
            if yend >= 0 and yend <= 1 and isinstance(yend, float):
                yend = int(yend * ysize)
            image_roi = image[ystart:yend, xstart:xend]

            # convert to grayscale
            if len(image.shape) > 2:
                image_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image_roi

            # mask of random value
            image_random = np.random.uniform(0, 1, size=(image_gray.shape[0], image_gray.shape[1]))

            gradient_direction = random.randint(self.line_gradient_direction[0], self.line_gradient_direction[1])

            # get gradients in horizontal and vertical direction
            gx, gy = np.gradient(image_gray, edge_order=1)

            # horizontal or both gradients
            if gradient_direction != 1:
                # remove negative values
                gx = abs(gx)

                # remove gradients beyond the selected range
                gx[gx <= self.line_gradient_range[0]] = 0
                gx[gx > self.line_gradient_range[1]] = 0

                # randomly remove line value
                gx[image_random < line_split_probability] = 0
                # get contours of lines
                contours_x, hierarchy = cv2.findContours(
                    gx.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE,
                )
                # get mask of lines
                mask_x = np.zeros_like(image_gray)
                for contour in contours_x:
                    x, y, w, h = cv2.boundingRect(contour)
                    # for horizontal line
                    if (
                        w > h * long_to_short_ratio
                        and w > line_min_length
                        and np.random.random() < line_replacement_probability
                    ):
                        cv2.drawContours(
                            mask_x,
                            contour,
                            -1,
                            (255, 255, 255),
                            random.randint(self.line_replacement_thickness[0], self.line_replacement_thickness[1]),
                        )

            # vertical or both gradients
            if gradient_direction != 0:
                # remove negative values
                gy = abs(gy)

                # remove gradients beyond the selected range
                gy[gy <= self.line_gradient_range[0]] = 0
                gy[gy > self.line_gradient_range[1]] = 0

                # randomly remove line value
                gy[image_random < line_split_probability] = 0
                # get contours of lines
                contours_y, hierarchy = cv2.findContours(
                    gy.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE,
                )
                # get mask of lines
                mask_y = np.zeros_like(image_gray)
                for contour in contours_y:
                    x, y, w, h = cv2.boundingRect(contour)
                    # for vertical line
                    if (
                        h > w * long_to_short_ratio
                        and h > line_min_length
                        and np.random.random() < line_replacement_probability
                    ):
                        cv2.drawContours(
                            mask_y,
                            contour,
                            -1,
                            (255, 255, 255),
                            random.randint(self.line_replacement_thickness[0], self.line_replacement_thickness[1]),
                        )

            # merge mask and set max value = 1
            if gradient_direction == 2:
                mask_xy = mask_x + mask_y
            elif gradient_direction == 1:
                mask_xy = mask_y
            else:
                mask_xy = mask_x
            mask_xy[mask_xy > 0] = 1

            # output image
            image_output = image.copy()

            # mask with replacement value
            replacement_mask = np.random.randint(
                self.line_replacement_value[0],
                self.line_replacement_value[1] + 1,
                size=(yend - ystart, xend - xstart),
            )

            # replace detected lines with line value
            if len(image_output.shape) > 2:
                # skip alpha layer
                for i in range(3):
                    image_output[ystart:yend, xstart:xend, i][mask_xy > 0] = replacement_mask[mask_xy > 0]
            else:
                image_output[ystart:yend, xstart:xend][mask_xy > 0] = replacement_mask[mask_xy > 0]

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

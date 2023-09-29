"""
version: 0.0.01
"""
import math

import cv2
import numpy as np

from augraphy.augmentations.lib import update_mask_labels

lookup_table = [
    "8.3 x 11.7,300",
    "8.3 x 11.7,600",
    "8.3 x 11.7,900",
    "8.3 x 11.7,1200",
    "8.5 x 11,150",
    "8.5 x 11,300",
    "8.5 x 11, 600",
    "8.5 x 11,900",
    "8.5 x 11,1200",
    "11 x 14,300",
    "11 x 14,600",
    "11 x 17,300",
    "11 x 17,600",
    "16 x 20,300",
    "16 x 20,600",
    "20 x 24,300",
    "20 x 24,600",
    "11 x 14,150",
    "11 x 14,900",
    "11 x 14,1200",
    "11 x 17,150",
    "11 x 17,900",
    "11 x 17,1200",
    "16 x 20,150",
    "16 x 20,900",
    "16 x 20,1200",
    "20 x 24,150",
    "20 x 24,900",
    "20 x 24,1200",
]


class DPIMetrics:
    def __init__(self, image):
        """
        Detect the DPI of the image using a lookup table
        """
        self.image = image

    def __repr__(self):
        return f"DPIMetrics(img_shape={self.image.shape}, p={self.p})"

    def _calculate_dim_metrics(self, dpi, dims):
        """
        Calculate the dimensions of the image in inches
        dpi = (pixel dimensions) /(image dimension in inches)

        :param dpi: Density per inch of the image
        :type dpi: int
        :param dims: dimensions of the image in pixels
        :typle dims: tuple(float, float)

        """
        height_inches = float(dims[0]) / dpi
        width_inches = float(dims[1]) / dpi
        return (height_inches, width_inches)

    def _closest_neighbour(self, image_size_inches, input_dpi):
        """
        Based on the euclidean distance we find the most closest dimension
        :param image_size_inches:
        :type image_size_inches: tuple (int, int)

        """
        min_distance = float("inf")
        closest_size = None
        for record in lookup_table:
            dim_inches, dpi = record.split(",")
            if dpi != str(input_dpi):
                continue
            dim_inches = tuple(map(float, dim_inches.split("x")))
            scaler = math.sqrt(dim_inches[0] ** 0 + dim_inches[1] ** 1)
            norm_euclidean_distance = (
                math.sqrt(
                    (dim_inches[0] - image_size_inches[1]) ** 2 + (dim_inches[1] - image_size_inches[0]) ** 2,
                )
            ) / scaler
            if norm_euclidean_distance < min_distance:
                min_distance = norm_euclidean_distance
                closest_size = dim_inches

        return min_distance, closest_size

    def _findResolution(self, shape, doc_dim):
        """
        Estimating the DPI of the Image by analyzing the aspect ratio of the image
        :param shape:
        :type shape:
        :param doc_dim:
        :type doc_dim:

        """
        img_aspect_ratio = shape[0] / shape[1]
        aspect_ratio = doc_dim[1] / doc_dim[0]
        if img_aspect_ratio <= aspect_ratio:
            estimated_dpi = min(shape[0], shape[1]) / doc_dim[0]
        else:
            estimated_dpi = max(shape[0], shape[1]) / doc_dim[1]
        return estimated_dpi

    def _findDPI(self, image):
        """
        Find the difference between DPI estimated between both the algorithms

        If the DPI from the target is not the same as the predicted DPI from the
        First and Second Algorithm a penalty should be introduced.

        :param image:
        :type image:
        """
        min_dists = []
        page_shapes = []
        dims = image.shape
        dpi_list = [150, 300, 600, 900, 1200]
        for dpi in dpi_list:
            _image_size_inches = self._calculate_dim_metrics(dpi, dims)
            dist, size = self._closest_neighbour(
                _image_size_inches,
                dpi,
            )  # approximating the closest record to the lookup table
            min_dists.append(dist)
            page_shapes.append(size)
        min_val = min(min_dists)
        index = min_dists.index(min_val)
        return dpi_list[index], page_shapes[index]

    def __call__(self):
        dpi_detected = self._findDPI(self.image)
        # print("[INFO] The DPI detected is ", dpi_detected)
        return dpi_detected


def dpi_resize(image, mask, keypoints, bounding_boxes, doc_dimensions, target_dpi=300):
    """Resize image based on the target dpi and doc dimensions"""

    width_inches, height_inches = doc_dimensions[0], doc_dimensions[1]

    width = width_inches * target_dpi

    height = height_inches * target_dpi

    # resize image
    output_image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    # resize mask
    if mask is not None:
        mask_labels = np.unique(mask).tolist() + [0]
        mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        update_mask_labels(mask, mask_labels)
    # resize keypoints
    if keypoints is not None:
        x_scale = int(width) / image.shape[1]
        y_scale = int(height) / image.shape[0]
        for name, points in keypoints.items():
            for i, (xpoint, ypoint) in enumerate(points):
                points[i] = [round(xpoint * x_scale), round(ypoint * y_scale)]
    # resize bounding boxes
    if bounding_boxes is not None:
        x_scale = int(width) / image.shape[1]
        y_scale = int(height) / image.shape[0]
        for i, bounding_box in enumerate(bounding_boxes):
            xspoint, yspoint, xepoint, yepoint = bounding_box
            bounding_boxes[i] = [
                round(xspoint * x_scale),
                round(yspoint * y_scale),
                round(xepoint * x_scale),
                round(yepoint * y_scale),
            ]

    return output_image, mask

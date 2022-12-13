import random

import cv2
import numpy as np
from sklearn.datasets import make_blobs

from augraphy.base.augmentation import Augmentation


class Letterpress(Augmentation):
    """Produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

    :param n_samples: Pair of ints determining number of points in a cluster.
    :type n_samples: tuple, optional
    :param n_clusters: Pair of ints determining number of clusters.
    :type n_clusters: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :param value_threshold_range: Min value of pixel to enable letterpress effect.
    :type value_threshold_range: tuple, optional
    :param blur: Flag to enable blur in letterpress noise mask.
    :type blur: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        n_samples=(300, 800),
        n_clusters=(300, 800),
        std_range=(1500, 5000),
        value_range=(200, 255),
        value_threshold_range=(128, 128),
        blur=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.std_range = std_range
        self.value_range = value_range
        self.value_threshold_range = value_threshold_range
        self.blur = blur

    def __repr__(self):
        return f"Letterpress(n_samples={self.n_samples}, std_range={self.std_range}, value_range={self.value_range}, value_threshold_range={self.value_threshold_range}, blur={self.blur}, p={self.p})"

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            ysize, xsize = image.shape[:2]
            max_box_size = max(ysize, xsize)

            noise_mask = np.copy(image)

            generated_points = np.array([[-1, -1]], dtype="float")

            for i in range(random.randint(8, 12)):

                n_samples = [
                    random.randint(self.n_samples[0], self.n_samples[1])
                    for _ in range(random.randint(self.n_clusters[0], self.n_clusters[1]))
                ]
                std = random.randint(self.std_range[0], self.std_range[1]) / 100

                # generate clusters of blobs
                generated_points_new, point_group = make_blobs(
                    n_samples=n_samples,
                    center_box=(0, max_box_size),
                    cluster_std=std,
                    n_features=2,
                )

                generated_points = np.concatenate((generated_points, generated_points_new), axis=0)

            # remove decimals
            generated_points = generated_points.astype("int")

            # delete location where < 0 and > image size
            ind_delete = np.logical_or.reduce(
                (
                    generated_points[:, 0] < 0,
                    generated_points[:, 1] < 0,
                    generated_points[:, 0] > xsize - 1,
                    generated_points[:, 1] > ysize - 1,
                ),
            )
            generated_points_x = np.delete(generated_points[:, 0], ind_delete.reshape(ind_delete.shape[0]), axis=0)
            generated_points_y = np.delete(generated_points[:, 1], ind_delete.reshape(ind_delete.shape[0]), axis=0)

            # initialize empty noise mask and noise mask with random values
            noise_mask = np.zeros_like(image, dtype="uint8")
            if len(image.shape) > 2:
                noise_mask2 = (np.random.random((image.shape[0], image.shape[1], image.shape[2])) * 255).astype("uint8")
            else:
                noise_mask2 = (np.random.random((image.shape[0], image.shape[1])) * 255).astype("uint8")
            # generate random values in value range
            min_array_value = np.min(noise_mask2)
            max_array_value = np.max(noise_mask2)
            ratio = (self.value_range[1] - self.value_range[0]) / (max_array_value - min_array_value)
            # scale random value within range
            noise_mask2 = (ratio * noise_mask2) + (self.value_range[0] - (ratio * min_array_value))

            # insert noise value according to generate points
            if len(image.shape) > 2:
                noise_mask[generated_points_y, generated_points_x, :] = noise_mask2[
                    generated_points_y,
                    generated_points_x,
                    :,
                ]
            else:
                noise_mask[generated_points_y, generated_points_x] = noise_mask2[generated_points_y, generated_points_x]

            if self.blur:
                # gaussian blur needs uint8 input
                noise_mask = cv2.GaussianBlur(noise_mask, (5, 5), 0)

            if self.value_threshold_range[1] >= self.value_threshold_range[0]:
                value_threshold = random.randint(self.value_threshold_range[0], self.value_threshold_range[1])
            else:
                value_threshold = self.value_threshold_range[1]

            # apply noise to image
            indices = image < value_threshold
            image[indices] = noise_mask[indices]

            return image

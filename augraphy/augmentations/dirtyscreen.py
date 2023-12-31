import random

import cv2
import numpy as np
from sklearn.datasets import make_blobs

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder


class DirtyScreen(Augmentation):
    """Creates a dirty screen effect by applying blobs of different size in the entire image.

    :param n_clusters: Pair of ints determining number of clusters.
        Each cluster represents a dirty area.
    :type n_clusters: tuple, optional
    :param n_samples: Pair of ints determining number of points in a cluster.
        Higher point value contributes to a denser dirty area.
    :type n_samples: tuple, optional
    :param std_range: Pair of ints determining the size of a cluster.
    :type std_range: tuple, optional
    :param value_range: Pair of ints determining the value of the points in a the cluster.
    :type value_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        n_clusters=(50, 100),
        n_samples=(2, 20),
        std_range=(1, 5),
        value_range=(150, 250),
        p=1,
    ):
        super().__init__(p=p)
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.std_range = std_range
        self.value_range = value_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"DirtyScreen(n_clusters={self.n_clusters}, n_samples={self.n_samples}, std_range={self.std_range}, value_range={self.value_range}, p={self.p})"

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

            noise_mask_merged = np.full_like(image, fill_value=255, dtype="uint8")

            for i in range(10):

                max_box_size = max(ysize, xsize)

                center_box_min = 0
                center_box_max = max_box_size - 1

                n_samples = [
                    random.randint(self.n_samples[0], self.n_samples[1])
                    for _ in range(random.randint(self.n_clusters[0], self.n_clusters[1]))
                ]
                std = random.randint(self.std_range[0], self.std_range[1])

                # generate clusters of blobs
                generated_points, point_group = make_blobs(
                    n_samples=n_samples,
                    center_box=(center_box_min, center_box_max),
                    cluster_std=std,
                    n_features=2,
                )

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

                # initialize empty noise mask and noise mask with value range
                noise_mask = np.full_like(image, fill_value=255, dtype="uint8")
                noise_mask_value = np.random.randint(
                    self.value_range[0],
                    self.value_range[1],
                    size=(ysize, xsize),
                    dtype="uint8",
                )

                # insert noise value according to generate points
                if len(image.shape) > 2:
                    # skip alpha layer
                    for i in range(3):
                        noise_mask[generated_points_y, generated_points_x, i] = noise_mask_value[
                            generated_points_y,
                            generated_points_x,
                        ]
                else:
                    noise_mask[generated_points_y, generated_points_x] = noise_mask_value[
                        generated_points_y,
                        generated_points_x,
                    ]

                # merge points
                noise_mask_merged = cv2.multiply(noise_mask_merged, noise_mask, scale=1 / 255)

            # add points into input
            image_output = image.copy()
            image_output[noise_mask_merged != 255] = noise_mask_merged[noise_mask_merged != 255]

            # blend points image into input again
            ob = OverlayBuilder(
                "ink_to_paper",
                image_output,
                image,
                1,
                (1, 1),
                "center",
                0,
                0.5,
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

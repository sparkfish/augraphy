import numpy as np
import random

from sklearn.datasets import make_blobs

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class LowInkBlobsAugmentation(Augmentation):
    """Creates random blobs of "low ink" to apply to the image.

    :param count_range: Pair of ints determining the range from which the number
           of blobs to generate is sampled.
    :type count_range: tuple, optional
    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :param probability: The probability this Augmentation will be applied.
    :type probability: float, optional
    """

    def __init__(
        self,
        count_range=(5, 25),
        size_range=(10, 20),
        points_range=(5, 25),
        std_range=(10, 75),
        features_range=(15, 25),
        value_range=(180, 250),
        probability=0.5,
    ):
        """Constructor method"""
        super().__init__(probability=probability)
        self.count_range = count_range
        self.size_range = size_range
        self.points_range = points_range
        self.std_range = std_range
        self.features_range = features_range
        self.value_range = value_range
        apply = lambda x, y: x if x > y else y
        self.apply = np.vectorize(apply)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkBlobsAugmentation(count_range={self.count_range}, size_range={self.size_range}, points_range={self.points_range}, std_range={self.std_range}, features_range={self.features_range}, value_range={self.value_range}, probability={self.probability})"

    def create_blob(self):
        """Generates a Gaussian blob to place in the image."""
        size = random.randint(self.size_range[0], self.size_range[1])
        std = random.randint(self.std_range[0], self.std_range[1]) / 100
        points = random.randint(self.points_range[0], self.points_range[1])
        features = random.randint(self.features_range[0], self.features_range[1])

        X, y = make_blobs(
            n_samples=points, cluster_std=[std], centers=[(0, 0)], n_features=features
        )  # , random_state=1)
        X *= size // 4
        X += size // 2
        X = [[int(item) for item in items] for items in X]
        blob = np.full((size, size, 1), 0, dtype="uint8")

        for point in X:
            if (
                point[0] < blob.shape[0]
                and point[1] < blob.shape[1]
                and point[0] > 0
                and point[1] > 0
            ):
                value = random.randint(self.value_range[0], self.value_range[1])
                blob[point[0], point[1]] = value

        return blob

    def apply_blob(self, mask):
        """Places a Gaussian blob at a random location in the image."""
        blob = self.create_blob()
        x_start = random.randint(0, mask.shape[1] - blob.shape[1])
        y_start = random.randint(0, mask.shape[0] - blob.shape[0])
        x_stop = x_start + blob.shape[1]
        y_stop = y_start + blob.shape[0]
        mask_chunk = mask[y_start:y_stop, x_start:x_stop]
        mask[y_start:y_stop, x_start:x_stop] = self.apply(mask_chunk, blob[:, :, 0])

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["ink"][-1].result.copy()
            count = random.randint(self.count_range[0], self.count_range[1])

            for i in range(count):
                self.apply_blob(image)

            data["ink"].append(AugmentationResult(self, image))

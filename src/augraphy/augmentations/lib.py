"""This module contains functions generally useful for building augmentations."""

import numpy as np
from sklearn.datasets import make_blobs
import random

def addNoise(image, intensity_range=(0.1, 0.2), color_range=(0, 224)):
    """Applies random noise to the input image.

    :param image: The image to noise.
    :type image: numpy.array
    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    """

    intensity = random.uniform(intensity_range[0], intensity_range[1])
    noise = (
        lambda x: random.randint(color_range[0], color_range[1])
        if (x == 0 and random.random() < intensity)
        else x
    )
    add_noise = np.vectorize(noise)

    return add_noise(image)


def _create_blob(
    size_range=(10, 20),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
):
    """Generates a Gaussian noise blob for placement in an image.
    To be used with _apply_blob()

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
    """
    size = random.randint(size_range[0], size_range[1])
    std = random.randint(std_range[0], std_range[1]) / 100
    points = random.randint(points_range[0], points_range[1])
    features = random.randint(features_range[0], features_range[1])

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
            value = random.randint(value_range[0], value_range[1])
            blob[point[0], point[1]] = value

    return blob


def applyBlob(
    image,
    size_range=(10, 20),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
):
    """Places a Gaussian blob at a random location in the image.

    :param image: The image to place the blob in.
    :type image: numpy.array
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
    """
    blob = _create_blob(
        size_range, points_range, std_range, features_range, value_range
    )

    x_start = random.randint(0, image.shape[1] - blob.shape[1])
    y_start = random.randint(0, image.shape[0] - blob.shape[0])
    x_stop = x_start + blob.shape[1]
    y_stop = y_start + blob.shape[0]

    image_chunk = image[y_start:y_stop, x_start:x_stop]

    apply_chunk = np.vectorize(lambda x, y: max(x, y))

    image[y_start:y_stop, x_start:x_stop] = apply_chunk(image_chunk, blob[:, :, 0])

    return image

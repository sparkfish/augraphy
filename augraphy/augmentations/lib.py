"""This module contains functions generally useful for building augmentations."""
import random

import cv2
import numpy as np
from sklearn.datasets import make_blobs


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
    noise = lambda x: random.randint(color_range[0], color_range[1]) if (x == 0 and random.random() < intensity) else x
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
        n_samples=points,
        cluster_std=[std],
        centers=[(0, 0)],
        n_features=features,
    )  # , random_state=1)
    X *= size // 4
    X += size // 2
    X = [[int(item) for item in items] for items in X]
    blob = np.full((size, size, 1), 0, dtype="uint8")

    for point in X:
        if point[0] < blob.shape[0] and point[1] < blob.shape[1] and point[0] > 0 and point[1] > 0:
            value = random.randint(value_range[0], value_range[1])
            blob[point[0], point[1]] = value

    return blob


def applyBlob(
    mask,
    size_range=(10, 20),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
):
    """Places a Gaussian blob at a random location in the image.

    :param mask: The image to place the blob in.
    :type mask: numpy.array
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
    dim = min(
        mask.shape[0],
        mask.shape[1],
    )  # we don't want to generate blobs larger than the mask

    # temporary local variables, in case
    size = size_range
    std = std_range

    # make sure we don't generate a blob larger than the mask
    if 2 * (size_range[1] + std_range[1]) > dim:
        # don't make a radius that won't fit in our mask
        size = (1, dim // 2 - 1)
        # don't make a std.deviation that when added to radius, is larger than mask
        std = (0, dim // 2 - size[1])

    if size[1] < size[0]:
        return mask  # solve error where size[1] is smaller than size[0]

    blob = _create_blob(size, points_range, std, features_range, value_range)

    x_start = random.randint(0, mask.shape[1] - blob.shape[1])
    y_start = random.randint(0, mask.shape[0] - blob.shape[0])
    x_stop = x_start + blob.shape[1]
    y_stop = y_start + blob.shape[0]

    mask_chunk = mask[y_start:y_stop, x_start:x_stop]

    apply_chunk = np.vectorize(lambda x, y: max(x, y))

    mask_dim = len(mask.shape)  # mask channels
    if mask_dim > 2:  # colour image or > 3 channels
        for i in range(mask_dim):
            mask[y_start:y_stop, x_start:x_stop, i] = apply_chunk(
                mask_chunk[:, :, i],
                blob[:, :, 0],
            )
    else:  # single channel grayscale or binary image
        mask[y_start:y_stop, x_start:x_stop] = apply_chunk(
            mask_chunk[:, :],
            blob[:, :, 0],
        )

    return mask


def binaryThreshold(
    img,
    enable_otsu=0,
    enable_simple=0,
    simple_method=cv2.THRESH_BINARY,
    thres=127,
    max_value=255,
    enable_adaptive=0,
    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    block_size=21,
    C=10,
):
    """Converts img to grayscale and applies the given threshold type

    :param img: the image to binarize
    :type img: numpy.array

    :param enable_otsu: Flag to enable otsu binarization method
    :type enable_otsu: Int, optional

    :param enable_simple: Flag to enable simple binarization method
    :type enable_simple: Int, optional
    :param simple_method: Simple binarization method
    :type simple_method: cv2.THRESH_ value, optional
    :param thres: Thresholding value for simple binarization method
    :type thres: Int, optional
    :param max_value: Max value of image for simple binarization method
    :type max_value: Int, optional

    :param enable_adaptive: Flag to enable adaptive binarization method
    :type enable_adaptive: Int, optional
    :param adaptive_method: Adaptive binarization method
    :type adaptive_method: cv2.THRESH_ value, optional
    :param block_size: Size of a pixel neighborhood that is used to calculate
                        a threshold value for adaptive method
    :type block_size: Int, optional
    :param C: Constant subtracted from the mean or weighted mean, only for
                        adaptive method
    :type C: Int, optional
    """

    # otsu as default method
    if not enable_otsu and not enable_simple and not enable_adaptive:
        enable_otsu = 1

    # convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # otsu method:
    if enable_otsu:
        ret, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    # simple method
    elif enable_simple:
        simple_methods = [
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV,
        ]
        if simple_method in simple_methods:
            ret, thresholded = cv2.threshold(grayscale, thres, max_value, simple_method)
        else:
            raise TypeError("Invalid thresholding method.")
    # adaptive method
    elif enable_adaptive:
        adaptive_methods = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
        if adaptive_method in adaptive_methods:
            thresholded = cv2.adaptiveThreshold(
                grayscale,
                255,
                adaptive_method,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
        else:
            raise TypeError("Invalid thresholding method.")

    return thresholded


def sobel(image):
    """Computes the gradient of the image intensity function.

    :param image: The image over which to create an edge mask.
    :type image: numpy.array
    """
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient


def make_white_transparent(img, ink_color=0):
    # Create the Ink Layer for the specified color.
    img_bgra = cv2.cvtColor(
        np.full((img.shape[0], img.shape[1], 3), ink_color, dtype="uint8"),
        cv2.COLOR_BGR2BGRA,
    )

    # Convert to grayscale if not already.
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img.astype(np.single), cv2.COLOR_BGR2GRAY)

    # Apply transparency mask based on grayscale.
    img_bgra[:, :, 3] = ~(img[:, :].astype(np.int64))
    return img_bgra

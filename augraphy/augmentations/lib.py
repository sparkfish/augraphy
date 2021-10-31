"""This module contains functions generally useful for building augmentations."""
import random

import cv2
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import inv
from sklearn.datasets import make_blobs


def chaikin(points):
    """

    :param points: a list of more than 2 points, where each point is a tuple/array of len=2
    :type points: array
    """
    # appending the first point in path
    path = [points[0]]
    percent = 0.25
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        # distance between x values of two subsequent points
        dx = p1[0] - p0[0]
        # distance between y values of two subsequent points
        dy = p1[1] - p0[1]
        # creating two new points having 25% and 75% distance from the previous point
        new_p0 = (p0[0] + dx * percent, p0[1] + dy * percent)
        new_p1 = (p0[0] + dx * (1 - percent), p0[1] + dy * (1 - percent))
        path.append(new_p0)
        path.append(new_p1)
    # appending last point in path list
    path.append(points[-1])
    return path


def smooth(points, iter):
    """

    :param points: a list of more than 2 points, where each point is a tuple/array of len=2
    :type points: array
    :param iter: number of times to apply chaikin algorithm
    :type iter: int
    :return:
    """
    for i in range(iter):
        points = chaikin(points)
    return points


def add_noise(image, intensity_range=(0.1, 0.2), color_range=(0, 224)):
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


def apply_blob(
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


def binary_threshold(
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
    if len(img.shape) > 2:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = img

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


def mixed_blend_crude(img_source, img_background):
    """Blend 2 images using mixed gradients blend, very crude but fast"""

    # get background size
    ysize, xsize = img_background.shape[:2]

    # get source size
    sysize, sxsize = img_source.shape[:2]

    # if source size is different, resize it
    if sysize != ysize or sxsize != xsize:
        img_source = cv2.resize(img_source, (xsize, ysize), interpolation=cv2.INTER_AREA)

    # get source image channels
    if len(img_source) > 2:
        source_channel = img_source.shape[2]
    else:
        source_channel = 0

    # get background image channels
    if len(img_background) > 2:
        background_channel = img_background.shape[2]
    else:
        background_channel = 0

    img_source = img_source.astype("int")
    img_background = img_background.astype("int")
    img_outcome = img_background.copy().astype("int")

    # for shifting
    y_shifts = [0, 0, 1, -1]
    x_shifts = [1, -1, 0, 0]

    for y in range(1, ysize - 1, 1):
        for x in range(1, xsize - 1, 1):
            for x_shift, y_shift in zip(x_shifts, y_shifts):

                # source
                # multiple channels
                if source_channel:
                    for i in range(source_channel):
                        source_difference = img_source[y, x, i] - img_source[y + y_shift, x + x_shift, i]
                # single channel
                else:
                    source_difference = img_source[y, x] - img_source[y + y_shift, x + x_shift]

                # background
                # multiple channels
                if background_channel:
                    for i in range(background_channel):
                        background_difference = img_background[y, x, i] - img_background[y + y_shift, x + x_shift, i]
                # single channel
                else:
                    background_difference = img_background[y, x] - img_background[y + y_shift, x + x_shift]

                # output
                # dominant source difference
                if abs(source_difference) > abs(background_difference):
                    if source_channel:
                        for i in range(source_channel):
                            img_outcome[y, x, i] += source_difference
                    else:
                        img_outcome[y, x] += source_difference
                # dominant background difference
                else:
                    if source_channel:
                        for i in range(source_channel):
                            img_outcome[y, x, i] += background_difference
                    else:
                        img_outcome[y, x] += background_difference

    img_outcome[img_outcome > 255] = 255
    img_outcome[img_outcome < 0] = 0
    img_outcome = img_outcome.astype("uint8")

    return img_outcome


def mixed_blend_colour(img_source, img_background):
    """Blend 2 images using precise mixed gradients blending approach"""

    # get background gray
    if len(img_background.shape) > 2:
        img_background_gray = cv2.cvtColor(img_background, cv2.COLOR_BGR2GRAY)
    else:
        img_background_gray = img_background

    # get source gray
    if len(img_source.shape) > 2:
        img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    else:
        img_source_gray = img_source

    if np.sum(img_background_gray - img_source_gray) < 10:
        return img_background

    # scale to 0-1
    img_source = (1 / 255) * img_source.astype("float")
    img_background = (1 / 255) * img_background.astype("float")

    # get all pixels location
    y_loc, x_loc = np.where(img_background_gray >= 0)

    # get background size
    ysize, xsize = img_background.shape[:2]

    # get source size
    sysize, sxsize = img_source.shape[:2]

    # number of variables to be solved
    var = ysize * xsize

    # matrix that maps each pixel to a variable number
    im2var = np.zeros((ysize, xsize)).astype("int")
    i = 0
    for j in range(var):
        im2var[y_loc[j], x_loc[j]] = i
        i += 1

    img_blend = img_background.copy()

    # initialization
    img_outcome = np.zeros_like(img_background).astype("float")
    #    img_A = np.zeros((var, var))
    img_A = lil_matrix((var, var))  # sparse version

    # background
    # top
    img_background_top_shift = np.zeros_like(img_background).astype("float")
    img_background_top_shift[1:, :] = img_background[1:, :] - img_background[:-1, :]
    # bottom
    img_background_bottom_shift = np.zeros_like(img_background).astype("float")
    img_background_bottom_shift[:-1, :] = img_background[:-1, :] - img_background[1:, :]
    # left
    img_background_left_shift = np.zeros_like(img_background).astype("float")
    img_background_left_shift[:, 1:] = img_background[:, 1:] - img_background[:, :-1]
    # right
    img_background_right_shift = np.zeros_like(img_background).astype("float")
    img_background_right_shift[:, :-1] = img_background[:, :-1] - img_background[:, 1:]

    # source
    # top
    img_source_top_shift = np.zeros_like(img_background).astype("float")
    img_source_top_shift[1:, :] = img_source[1:, :] - img_source[:-1, :]
    # bottom
    img_source_bottom_shift = np.zeros_like(img_background).astype("float")
    img_source_bottom_shift[:-1, :] = img_source[:-1, :] - img_source[1:, :]
    # left
    img_source_left_shift = np.zeros_like(img_background).astype("float")
    img_source_left_shift[:, 1:] = img_source[:, 1:] - img_source[:, :-1]
    # right
    img_source_right_shift = np.zeros_like(img_background).astype("float")
    img_source_right_shift[:, :-1] = img_source[:, :-1] - img_source[:, 1:]

    # top
    top_background_dominant = np.where(abs(img_background_top_shift) > abs(img_source_top_shift))
    top_source_dominant = np.where(abs(img_source_top_shift) > abs(img_background_top_shift))
    img_outcome_tem1 = np.zeros_like(img_background).astype("float")
    img_outcome_tem2 = np.zeros_like(img_background).astype("float")
    img_outcome_tem1[top_source_dominant[:2]] += img_source_top_shift[top_source_dominant[:2]]
    img_outcome_tem2[top_background_dominant[:2]] += img_background_top_shift[top_background_dominant[:2]]
    img_outcome_tem1_sum = np.sum(abs(img_outcome_tem1), 2)
    img_outcome_tem2[img_outcome_tem1_sum != 0] = 0
    img_outcome += img_outcome_tem1 + img_outcome_tem2

    # bottom
    bottom_background_dominant = np.where(abs(img_background_bottom_shift) > abs(img_source_bottom_shift))
    bottom_source_dominant = np.where(abs(img_source_bottom_shift) > abs(img_background_bottom_shift))
    img_outcome_tem1 = np.zeros_like(img_background).astype("float")
    img_outcome_tem2 = np.zeros_like(img_background).astype("float")
    img_outcome_tem1[bottom_source_dominant[:2]] += img_source_bottom_shift[bottom_source_dominant[:2]]
    img_outcome_tem2[bottom_background_dominant[:2]] += img_background_bottom_shift[bottom_background_dominant[:2]]
    img_outcome_tem1_sum = np.sum(abs(img_outcome_tem1), 2)
    img_outcome_tem2[img_outcome_tem1_sum != 0] = 0
    img_outcome += img_outcome_tem1 + img_outcome_tem2

    # left
    left_background_dominant = np.where(abs(img_background_left_shift) > abs(img_source_left_shift))
    left_source_dominant = np.where(abs(img_source_left_shift) > abs(img_background_left_shift))
    img_outcome_tem1 = np.zeros_like(img_background).astype("float")
    img_outcome_tem2 = np.zeros_like(img_background).astype("float")
    img_outcome_tem1[left_source_dominant[:2]] += img_source_left_shift[left_source_dominant[:2]]
    img_outcome_tem2[left_background_dominant[:2]] += img_background_left_shift[left_background_dominant[:2]]
    img_outcome_tem1_sum = np.sum(abs(img_outcome_tem1), 2)
    img_outcome_tem2[img_outcome_tem1_sum != 0] = 0
    img_outcome += img_outcome_tem1 + img_outcome_tem2

    # right
    right_background_dominant = np.where(abs(img_background_right_shift) > abs(img_source_right_shift))
    right_source_dominant = np.where(abs(img_source_right_shift) > abs(img_background_right_shift))
    img_outcome_tem1 = np.zeros_like(img_background).astype("float")
    img_outcome_tem2 = np.zeros_like(img_background).astype("float")
    img_outcome_tem1[right_source_dominant[:2]] += img_source_right_shift[right_source_dominant[:2]]
    img_outcome_tem2[right_background_dominant[:2]] += img_background_right_shift[right_background_dominant[:2]]
    img_outcome_tem1_sum = np.sum(abs(img_outcome_tem1), 2)
    img_outcome_tem2[img_outcome_tem1_sum != 0] = 0
    img_outcome += img_outcome_tem1 + img_outcome_tem2

    # convert to array
    img_outcome_array = img_outcome.reshape(ysize * xsize, 3)

    # get all coordinates
    all_y = np.concatenate(
        (
            top_background_dominant[0],
            top_source_dominant[0],
            bottom_background_dominant[0],
            bottom_source_dominant[0],
            left_background_dominant[0],
            left_source_dominant[0],
            right_background_dominant[0],
            right_source_dominant[0],
        ),
        axis=0,
    )

    all_x = np.concatenate(
        (
            top_background_dominant[1],
            top_source_dominant[1],
            bottom_background_dominant[1],
            bottom_source_dominant[1],
            left_background_dominant[1],
            left_source_dominant[1],
            right_background_dominant[1],
            right_source_dominant[1],
        ),
        axis=0,
    )

    # all index
    index_all = im2var[all_y, all_x]

    # shifted index
    # top
    all_y_minus_1 = all_y - 1
    all_x_y_minus_1 = all_x.copy()
    all_y_top = all_y.copy()
    all_x_top = all_x.copy()

    # bottom
    all_y_plus_1 = all_y + 1
    all_x_y_plus_1 = all_x.copy()
    all_y_bottom = all_y.copy()
    all_x_bottom = all_x.copy()

    # left
    all_x_minus_1 = all_x - 1
    all_y_x_minus_1 = all_y.copy()
    all_y_left = all_y.copy()
    all_x_left = all_x.copy()

    # right
    all_x_plus_1 = all_x + 1
    all_y_x_plus_1 = all_y.copy()
    all_y_right = all_y.copy()
    all_x_right = all_x.copy()

    # delete inappropriate index
    # shift top
    delete_y_minus_1 = np.where(all_y_minus_1 < 0)
    all_y_minus_1 = np.delete(all_y_minus_1, delete_y_minus_1)
    all_x_y_minus_1 = np.delete(all_x_y_minus_1, delete_y_minus_1)
    all_y_top = np.delete(all_y_top, delete_y_minus_1)
    all_x_top = np.delete(all_x_top, delete_y_minus_1)

    # shift bottom
    delete_y_plus_1 = np.where(all_y_plus_1 > ysize - 1)
    all_y_plus_1 = np.delete(all_y_plus_1, delete_y_plus_1)
    all_x_y_plus_1 = np.delete(all_x_y_plus_1, delete_y_plus_1)
    all_y_bottom = np.delete(all_y_bottom, delete_y_plus_1)
    all_x_bottom = np.delete(all_x_bottom, delete_y_plus_1)

    # shift left
    delete_x_minus_1 = np.where(all_x_minus_1 < 0)
    all_x_minus_1 = np.delete(all_x_minus_1, delete_x_minus_1)
    all_y_x_minus_1 = np.delete(all_y_x_minus_1, delete_x_minus_1)
    all_x_left = np.delete(all_x_left, delete_x_minus_1)
    all_y_left = np.delete(all_y_left, delete_x_minus_1)

    # shift right
    delete_x_plus_1 = np.where(all_x_plus_1 > xsize - 1)
    all_x_plus_1 = np.delete(all_x_plus_1, delete_x_plus_1)
    all_y_x_plus_1 = np.delete(all_y_x_plus_1, delete_x_plus_1)
    all_x_right = np.delete(all_x_right, delete_x_plus_1)
    all_y_right = np.delete(all_y_right, delete_x_plus_1)

    # get index of shifted
    # top
    index_y_minus_1 = im2var[all_y_minus_1, all_x_y_minus_1]
    index_x_for_y_minus_1 = im2var[all_y_top, all_x_top]
    # bottom
    index_y_plus_1 = im2var[all_y_plus_1, all_x_y_plus_1]
    index_x_for_y_plus_1 = im2var[all_y_bottom, all_x_bottom]
    # left
    index_x_minus_1 = im2var[all_y_x_minus_1, all_x_minus_1]
    index_y_for_x_minus_1 = im2var[all_y_left, all_x_left]
    # right
    index_x_plus_1 = im2var[all_y_x_plus_1, all_x_plus_1]
    index_y_for_x_plus_1 = im2var[all_y_right, all_x_right]

    # set value to -1 for shifted index
    img_A[index_all, index_all] = 4
    img_A[index_y_minus_1, index_x_for_y_minus_1] = -1
    img_A[index_y_plus_1, index_x_for_y_plus_1] = -1
    img_A[index_y_for_x_minus_1, index_x_minus_1] = -1
    img_A[index_y_for_x_plus_1, index_x_plus_1] = -1

    # reshape
    b_size_y, b_size_x = img_outcome_array.shape
    b1 = img_outcome_array[:, 0].reshape(b_size_y, 1)
    b2 = img_outcome_array[:, 1].reshape(b_size_y, 1)
    b3 = img_outcome_array[:, 2].reshape(b_size_y, 1)

    # solve ax=b, where x = a^-1 dot b
    inv_A = inv(csr_matrix(img_A))
    vb = inv_A.dot(b1)
    vg = inv_A.dot(b2)
    vr = inv_A.dot(b3)

    # scale min max
    vr = (vr - min(vr)) / (max(vr) - min(vr))
    vg = (vg - min(vg)) / (max(vg) - min(vg))
    vb = (vb - min(vb)) / (max(vb) - min(vb))

    # remove negative value
    vr[vr < 0] = 0
    vg[vg < 0] = 0
    vb[vb < 0] = 0

    # copy the values over to the target image to the area to be blended
    e = 0
    for i in range(var):
        y = y_loc[i]
        x = x_loc[i]

        img_blend[y, x, 0] = vb[e]
        img_blend[y, x, 1] = vg[e]
        img_blend[y, x, 2] = vr[e]
        e += 1

    return (img_blend * 255).astype("uint8")

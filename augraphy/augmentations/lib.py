"""This module contains functions generally useful for building augmentations."""
import random

import cv2
import numpy as np
from numpy.linalg import norm
from skimage.filters import threshold_li
from skimage.filters import threshold_local
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum
from skimage.filters import threshold_niblack
from skimage.filters import threshold_otsu
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_triangle
from skimage.filters import threshold_yen
from sklearn.datasets import make_blobs


def rotate_image(mat, angle):
    """Rotates an image (angle in degrees) and expands image to avoid
    cropping.
    """

    mat = cv2.bitwise_not(mat)
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    rotated_mat = cv2.bitwise_not(rotated_mat)
    return rotated_mat


# Generate average intensity value
def generate_average_intensity(image):
    # Adapted from this discussion
    # https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv/22020098#22020098
    if len(image.shape) > 2:
        # bgr image - create brightness with euclidean norm
        return np.average(norm(image, axis=2)) / np.sqrt(3)
    else:
        # grayscale or binary
        return np.average(image)


# Generate noise to edges of folding
def add_folding_noise(img, side, p=0.1):
    # side = flag to put more noise at certain side
    #   0  = left side
    #   1  = right side

    # get image dimension
    ysize, xsize = img.shape[:2]

    for y in range(ysize):
        for x in range(xsize):

            if side:  # more noise on right side
                p_score = (((x) / xsize) ** 3) * p  # non linear score with power
            else:  # more noise on left side
                p_score = (((xsize - x) / xsize) ** 3) * p  # non linear score with power

            if p_score > random.random():
                img[y, x] = 0
    return img


# Perspective transform based on 4 points
def four_point_transform(image, pts, dst, xs, ys):
    M = cv2.getPerspectiveTransform(pts, dst)
    if not isinstance(image, np.floating):
        image = image.astype("float")
    img_warped = cv2.warpPerspective(image, M, (xs, ys))
    # return the warped image
    return img_warped


# Transform left side of folding area
def warp_fold_left_side(
    img,
    ysize,
    fold_noise,
    fold_x,
    fold_width_one_side,
    fold_y_shift,
):

    img_fuse = img.copy()

    # 4 vectices of folding area
    xs = 0  # xleft
    xe = fold_width_one_side  # xright
    ys = 0  # ytop
    ye = ysize  # ybottom

    # before distortion
    top_left = [xs, ys]
    top_right = [xe, ys]
    bottom_left = [xs, ye]
    bottom_right = [xe, ye]

    # after distortion
    dtop_left = [xs, ys]
    dtop_right = [xe, ys + fold_y_shift]
    dbottom_left = [xs, ye]
    dbottom_right = [xe, ye + fold_y_shift]

    # image cropping points
    cxs = fold_x
    cxe = fold_x + fold_width_one_side
    cys = 0
    cye = ysize

    # points of folding area
    source_pts = np.array(
        [top_left, bottom_left, bottom_right, top_right],
        dtype=np.float32,
    )
    destination_pts = np.array(
        [dtop_left, dbottom_left, dbottom_right, dtop_right],
        dtype=np.float32,
    )

    # crop section of folding area
    img_crop = img[cys:cye, cxs:cxe]

    # get image dimension of cropped image
    if len(img_crop.shape) > 2:
        cysize, cxsize, cdim = img_crop.shape
    else:
        cysize, cxsize = img_crop.shape
        cdim = 2

    # warp folding area
    img_warped = four_point_transform(
        img_crop,
        source_pts,
        destination_pts,
        cxsize,
        cysize + fold_y_shift,
    )
    img_warped = add_folding_noise(img_warped, 1, fold_noise / 2)

    if cdim > 2:
        img_fuse[cys:cye, cxs:cxe, :] = img_warped[:-fold_y_shift, :, :]
    else:
        img_fuse[cys:cye, cxs:cxe] = img_warped[:-fold_y_shift, :]

    return img_fuse


# Transform right side of folding area
def warp_fold_right_side(
    img,
    ysize,
    fold_noise,
    fold_x,
    fold_width_one_side,
    fold_y_shift,
):

    img_fuse = img.copy()

    # 4 vectices of folding area
    xs = 0  # xleft
    xe = fold_width_one_side  # xright
    ys = 0  # ytop
    ye = ysize  # ybottom

    # before distortion
    top_left = [xs, ys]
    top_right = [xe, ys]
    bottom_left = [xs, ye]
    bottom_right = [xe, ye]

    # after distortion
    dtop_left = [xs, ys + (fold_y_shift)]
    dtop_right = [xe, ys]
    dbottom_left = [xs, ye + (fold_y_shift)]
    dbottom_right = [xe, ye]

    # image cropping points
    cxs = fold_x + fold_width_one_side
    cxe = fold_x + (fold_width_one_side * 2)
    cys = 0
    cye = ysize

    # points of folding area
    source_pts = np.array(
        [top_left, bottom_left, bottom_right, top_right],
        dtype=np.float32,
    )
    destination_pts = np.array(
        [dtop_left, dbottom_left, dbottom_right, dtop_right],
        dtype=np.float32,
    )

    # crop section of folding area
    img_crop = img[cys:cye, cxs:cxe]

    # get image dimension of cropped image
    if len(img_crop.shape) > 2:
        cysize, cxsize, cdim = img_crop.shape
    else:
        cysize, cxsize = img_crop.shape
        cdim = 2

    # warp folding area
    img_warped = four_point_transform(
        img_crop,
        source_pts,
        destination_pts,
        cxsize,
        cysize + fold_y_shift,
    )
    img_warped = add_folding_noise(img_warped, 0, fold_noise / 2)

    if cdim > 2:
        img_fuse[cys:cye, cxs:cxe, :] = img_warped[:-fold_y_shift, :, :]
    else:
        img_fuse[cys:cye, cxs:cxe] = img_warped[:-fold_y_shift, :]

    return img_fuse


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
    # inherit ink from input image
    if ink_color == -1:
        img_bgra = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2BGRA,
        )
    # use the provided ink color
    else:
        img_bgra = cv2.cvtColor(
            np.full((img.shape[0], img.shape[1], 3), ink_color, dtype="uint8"),
            cv2.COLOR_BGR2BGRA,
        )

    # Convert to grayscale if not already.
    if len(img.shape) > 2 and img.shape[2] > 1:
        img_alpha = cv2.cvtColor(img.astype(np.single), cv2.COLOR_BGR2GRAY)
    else:
        img_alpha = img

    # Apply transparency mask based on grayscale.
    img_bgra[:, :, 3] = ~(img_alpha[:, :].astype(np.int64))
    return img_bgra


def binary_threshold(
    image,
    threshold_method,
    threshold_arguments,
):

    # convert image to grascale
    if len(image.shape) > 2:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image

    # if return grayscale image if threshold method is grayscale
    if threshold_method == "grayscale":
        return grayscale

    if threshold_arguments:
        # get input arguments for threhold function
        input_arguments = ""
        for input_argument in threshold_arguments:
            # for string argument value
            if isinstance(threshold_arguments[input_argument], str):
                input_value = threshold_arguments[input_argument]
            # for non-string argument value
            else:
                input_value = str(threshold_arguments[input_argument])
            # merge argument name and their value
            input_arguments += "," + input_argument + "=" + input_value

        # apply binary function and get threshold
        binary_threshold = eval(threshold_method + "(grayscale" + input_arguments + ")")
    else:
        # apply binary function and get threshold
        binary_threshold = eval(threshold_method + "(grayscale)")

    # apply binary threshold
    if threshold_method == "cv2.threshold":
        binary_threshold, image_binary = binary_threshold
    else:
        image_binary = np.uint8((grayscale > binary_threshold) * 255)

    return image_binary

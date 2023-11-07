"""This module contains functions generally useful for building augmentations."""
import os
import random
from glob import glob

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit
from numpy.linalg import norm
from PIL import Image
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


def load_image_from_cache(random_image=0):
    """Load image from augraphy cache folder.

    :param random_image: Flag to load random image from cache folder.
        If it is not set, it loads the latest modified image.
    :type random_image: int
    """

    # path to foreground cache folder
    cache_folder_path = os.path.join(os.getcwd() + "/augraphy_cache/")
    cache_image_paths = glob(cache_folder_path + "*.png", recursive=True)

    # at least 1 image
    if len(cache_image_paths) > 0:
        if random_image:
            image_index = random.randint(0, len(cache_image_paths) - 1)
        else:
            modified_time = [os.path.getmtime(image_path) for image_path in cache_image_paths]
            image_index = np.argmax(modified_time)
        # get random image
        image = cv2.imread(cache_image_paths[image_index])

        return image
    else:
        return None


@jit(nopython=True, cache=True, parallel=True)
def rotate_point(xpoint, ypoint, xcenter, ycenter, angle):
    """Rotate point around an origin based on the provided angle in clockwise direction.

    :param xpoint: The x coordinate of input point.
    :type xpoint: int
    :param ypoint: The y coordinate of input point.
    :type ypoint: int
    :param xcenter: The x origin point.
    :type xcenter: int
    :param ycenter: The y origin point.
    :type ycenter: int
    :param angle: The angle of rotation in degree.
    :type angle: int
    """

    angle_radian = np.deg2rad(angle)

    rotated_xpoint = xcenter + (
        (np.cos(angle_radian) * (xpoint - xcenter)) - (np.sin(angle_radian) * (ypoint - ycenter))
    )
    rotated_ypoint = ycenter + (
        (np.sin(angle_radian) * (xpoint - xcenter)) + (np.cos(angle_radian) * (ypoint - ycenter))
    )

    return rotated_xpoint, rotated_ypoint


def rotate_keypoints(keypoints, xcenter, ycenter, x_offset, y_offset, angle):
    """Rotate keypoints around an origin based on the provided angle.

    :param keypoints: The input keypoints.
    :type keypoints: dictionary
    :param xcenter: The x origin point.
    :type xcenter: int
    :param ycenter: The y origin point.
    :type ycenter: int
    :param x_offset: The relative x offset after the rotation.
    :type x_offset: int
    :param y_offset: The relative y offset after the rotation.
    :type y_offset: int
    :param angle: The angle of rotation in degree.
    :type angle: int
    """

    # rotate each label
    for name, points in keypoints.items():
        for i, (xpoint, ypoint) in enumerate(points):
            # use - fold_angle because image are rotated anticlockwise
            rotated_xpoint, rotated_ypoint = rotate_point(xpoint, ypoint, xcenter, ycenter, angle)
            points[i] = [round(rotated_xpoint + x_offset), round(rotated_ypoint + y_offset)]


def rotate_bounding_boxes(bounding_boxes, xcenter, ycenter, x_offset, y_offset, angle):
    """Rotate bounding boxes around an origin based on the provided angle.

    :param bounding_boxes: The input bounding boxes.
    :type bounding_boxes: list
    :param xcenter: The x origin point.
    :type xcenter: int
    :param ycenter: The y origin point.
    :type ycenter: int
    :param x_offset: The relative x offset after the rotation.
    :type x_offset: int
    :param y_offset: The relative y offset after the rotation.
    :type y_offset: int
    :param angle: The angle of rotation in degree.
    :type angle: int
    """

    for i, bounding_box in enumerate(bounding_boxes):
        xspoint, yspoint, xepoint, yepoint = bounding_box
        width = xepoint - xspoint
        height = yepoint - yspoint
        # based on start point (x0, y0) only
        rotated_xspoint, rotated_yspoint = rotate_point(xspoint, yspoint, xcenter, ycenter, angle)
        # update box with rotated points
        bounding_boxes[i] = [
            round(rotated_xspoint + x_offset),
            round(rotated_yspoint + y_offset),
            round(rotated_xspoint + x_offset + width),
            round(rotated_yspoint + y_offset + height),
        ]


def update_mask_labels(mask, mask_labels):
    """Quantitize labels of current mask based on the input mask labels.

    :param mask: The input mask.
    :type mask: numpy array
    :param mask_labels: A list contains input labels.
    :type mask_labels: list
    """

    empty_indices = mask == 0
    new_mask_labels = np.unique(mask)
    for new_mask_label in new_mask_labels:
        # new interpolated value, replace with old nearest value
        if new_mask_label not in mask_labels:
            differences = [abs(new_mask_label - mask_label) for mask_label in mask_labels]
            min_index = np.argmin(differences)
            mask[mask == new_mask_label] = mask_labels[min_index]
    mask[empty_indices] = 0


def quilt_texture(image_texture, patch_size, patch_number_width, patch_number_height):
    """Generate new texture image by quilting patches of input image.

    :param image_texture: The input image texture.
    :type image_texture: numpy array
    :param patch_size: The size of each image patch.
    :type patch_size: int
    :param patch_number_width: The number of image patch in horizontal direction.
    :type patch_number_width: int
    :param patch_number_height: The number of image patch in vertical direction.
    :type patch_number_height: int
    """

    overlap = patch_size // 5

    # size of output
    ysize = (patch_number_height * patch_size) - (patch_number_height - 1) * overlap
    xsize = (patch_number_width * patch_size) - (patch_number_width - 1) * overlap

    # convert from gray to bgr
    is_gray = 0
    if len(image_texture.shape) < 3:
        is_gray = 1
        image_texture = cv2.cvtColor(image_texture, cv2.COLOR_GRAY2BGR)

    # output
    image_quilt = np.zeros((ysize, xsize, image_texture.shape[2]), dtype="uint8")

    # size of image texture
    ysize, xsize = image_texture.shape[:2]

    # hsv channel of texture
    image_hsv = cv2.cvtColor(image_texture, cv2.COLOR_BGR2HSV)

    # get a reference patch's hue, saturation and value

    n = 0
    while n < 10:
        y = np.random.randint(ysize - patch_size)
        x = np.random.randint(xsize - patch_size)
        # to prevent black or white blank image
        if (
            np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 2]) < 245
            and np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 2]) > 10
        ):
            break
        n += 1

    h_reference = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 0])
    s_reference = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 1])
    v_reference = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 2])
    offset = 10
    h_range = [h_reference - offset, h_reference + offset]
    s_range = [s_reference - offset, s_reference + offset]
    v_range = [v_reference - offset, v_reference + offset]

    # generate and apply random patch
    for i in range(patch_number_height):
        for j in range(patch_number_width):
            y = i * (patch_size - overlap)
            x = j * (patch_size - overlap)
            image_patch = get_random_patch(
                image_texture,
                image_hsv,
                patch_size,
                ysize,
                xsize,
                h_range,
                s_range,
                v_range,
            )
            image_quilt[y : y + patch_size, x : x + patch_size] = image_patch

    # smoothing
    image_quilt = cv2.medianBlur(image_quilt, ksize=11)

    # enhance contrast of texture
    image_quilt = enhance_contrast(image_quilt)

    # image follows input texture color channel
    if is_gray:
        image_quilt = cv2.cvtColor(image_quilt, cv2.COLOR_BGR2GRAY)

    return image_quilt


def get_random_patch(image_texture, image_hsv, patch_size, ysize, xsize, h_range, s_range, v_range):
    """Get patch of image from texture based on input hue, saturation and value range.

    :param image_texture: The input image texture.
    :type image_texture: numpy array
    :param image_hsv: The input image texture in HSV channel.
    :type image_hsv: numpy array
    :param patch_size: The size of each image patch.
    :type patch_size: int
    :param y_size: The height of image texture.
    :type y_size: int
    :param x_size: The width of image texture.
    :type x_size: int
    :param h_range: The range of reference hue values.
    :type h_range: tuple
    :param s_range: The range of reference saturation values.
    :type s_range: tuple
    :param v_range: The range of reference value values.
    :type v_range: tuple
    """

    n = 0
    y = np.random.randint(ysize - patch_size)
    x = np.random.randint(xsize - patch_size)
    image_patch = image_texture[y : y + patch_size, x : x + patch_size]

    # use a fixed number to prevent infinity loops
    while n < 10:

        y = np.random.randint(ysize - patch_size)
        x = np.random.randint(xsize - patch_size)

        # get mean of h, s and v channel of current patch
        h_mean = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 0])
        s_mean = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 1])
        v_mean = np.mean(image_hsv[y : y + patch_size, x : x + patch_size, 2])

        if (
            h_mean >= h_range[0]
            and h_mean < h_range[1]
            and s_mean >= s_range[0]
            and s_mean < s_range[1]
            and v_mean >= v_range[0]
            and v_mean < v_range[1]
        ):

            # get patch of image
            image_patch = image_texture[y : y + patch_size, x : x + patch_size]

            # apply gamma correction
            mid = np.mean(v_range) / 255
            gamma = np.log(mid * 255) / np.log(v_mean)
            image_patch = np.power(image_patch, gamma).clip(0, 255).astype(np.uint8)
            break

        n += 1

    return image_patch


# adapted from this link:
# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def enhance_contrast(image):
    """Enhance image contrast by applying clahe in L channel of image.

    :param image: The input image.
    :type image: numpy array
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    image_merge = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    enhanced_image = cv2.cvtColor(image_merge, cv2.COLOR_LAB2BGR)

    return enhanced_image


# Adapted from this link:
# # https://stackoverflow.com/questions/51646185/how-to-generate-a-paper-like-background-with-opencv
def generate_noise(xsize, ysize, channel, ratio=1, sigma=1):
    """Generate noises through normal distribution.

    :param xsize: The width of the generated noise.
    :type xsize: int
    :param ysize: The height of the generated noise.
    :type ysize: int
    :param ratio: The size of generated noise pattern.
    :type ratio: int
    :param sigma: The bounds of noise fluctuations.
    :type sigma: float
    """

    new_ysize = int(ysize / ratio)
    new_xsize = int(xsize / ratio)
    result = np.random.normal(0, sigma, (new_xsize, new_ysize, channel))
    if ratio != 1:
        result = cv2.resize(result, dsize=(xsize, ysize), interpolation=cv2.INTER_LINEAR)
    return result.reshape((ysize, xsize, channel))


def generate_texture(ysize, xsize, channel, value=255, sigma=1, turbulence=2):
    """Generate random texture through multiple iterations of noise addition.

    :param xsize: The width of the generated noise.
    :type xsize: int
    :param ysize: The height of the generated noise.
    :type ysize: int
    :param channel: The number of channel in the generated noise.
    :type channel: int
    :param value: The initial value of the generated noise.
    :type value: int
    :param sigma: The bounds of noise fluctuations.
    :type sigma: float
    :param turbulence: The value to define how quickly big patterns will be replaced with the small ones.
    :type turbulence: int

    """
    image_output = np.full((ysize, xsize, 1), fill_value=value, dtype="float")
    ratio = min(xsize, ysize)
    while ratio != 1:
        image_output += generate_noise(xsize, ysize, 1, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    image_output = np.clip(image_output, 0, 255)

    new_min = 32
    new_max = 255
    image_output = ((image_output - np.min(image_output)) / (np.max(image_output) - np.min(image_output))) * (
        new_max - new_min
    ) + new_min

    # convert to uint8
    image_output = np.uint8(image_output)

    # conver to color image
    if channel == 3:
        image_output = cv2.cvtColor(image_output, cv2.COLOR_GRAY2BGR)
    elif channel == 1:
        # remove additional channel for grayscale image
        image_output = image_output[:, :, 0]

    return image_output


def generate_broken_edge_texture(oxsize, oysize):
    """Generate a mask of broken edges based texture.

    :param oxsize: The width of the output mask.
    :type oxsize: int
    :param oysize: The height of the output mask.
    :type oysize: int
    """

    iysize, ixsize = 400, 400

    # mask of noise
    image_noise = np.full((iysize, ixsize), fill_value=255, dtype="uint8")

    # image center
    center_xs = [0, int(ixsize / 2), ixsize, int(ixsize / 2)]
    center_ys = [int(iysize / 2), 0, int(iysize / 2), iysize]

    noise_value = [128, 255]
    n_clusters = [300, 500]
    n_samples = [1080, 1280]
    stds = [int(ixsize / 4), int(iysize / 4)]

    n_samples = [
        random.randint(n_samples[0], n_samples[1]) for i in range(random.randint(n_clusters[0], n_clusters[1]))
    ]
    std = random.randint(stds[0], stds[1])

    generated_points_x = np.array([[0]], dtype=("float"))
    generated_points_y = np.array([[0]], dtype=("float"))

    for center_x, center_y in zip(center_xs, center_ys):

        # generate clusters of noises
        cgenerated_points_x, _ = make_blobs(
            n_samples=n_samples,
            center_box=(center_x, center_x),
            cluster_std=std,
            n_features=1,
        )

        # generate clusters of noises
        cgenerated_points_y, _ = make_blobs(
            n_samples=n_samples,
            center_box=(center_y, center_y),
            cluster_std=std,
            n_features=1,
        )

        generated_points_x = np.concatenate((generated_points_x, cgenerated_points_x), axis=0)
        generated_points_y = np.concatenate((generated_points_y, cgenerated_points_y), axis=0)

    # generate x and y points of noise
    generated_points_x = generated_points_x.astype("int")
    generated_points_y = generated_points_y.astype("int")

    # remove invalid points
    ind_delete_x1 = np.where(generated_points_x < 2)
    ind_delete_x2 = np.where(generated_points_x >= ixsize - 2)
    ind_delete_y1 = np.where(generated_points_y < 2)
    ind_delete_y2 = np.where(generated_points_y >= iysize - 2)

    ind_delete = np.concatenate(
        (ind_delete_x1, ind_delete_x2, ind_delete_y1, ind_delete_y2),
        axis=1,
    )
    generated_points_x = np.delete(generated_points_x, ind_delete, axis=0)
    generated_points_y = np.delete(generated_points_y, ind_delete, axis=0)

    # update noise value
    image_random = np.random.random((iysize, ixsize))
    image_random[image_random < noise_value[0] / 255] = 0
    image_random[image_random > noise_value[1] / 255] = 0
    image_random = (image_random * 255).astype("uint8")

    # update points with random value
    image_noise[generated_points_y, generated_points_x] = image_random[generated_points_y, generated_points_x]

    # apply blur
    image_noise = cv2.GaussianBlur(image_noise, (random.choice([7, 9, 11, 13]), random.choice([7, 9, 11, 13])), 0)

    # create edge texture
    image_edge_texture = image_noise + image_noise

    # resize to expected size
    image_edge_texture = cv2.resize(image_edge_texture, (oxsize, oysize), 0)

    return image_edge_texture


# adapted from this repository:
# https://github.com/NewYaroslav/strange_pattern
@jit(nopython=True, cache=True, parallel=True)
def generate_strange_texture(oxsize, oysize):
    """Generate a random strange texture.

    :param oxsize: The width of the output texture image.
    :type oxsize: int
    :param oysize: The height of the output texture image.
    :type oysize: int
    """

    background_value = random.uniform(0.04, 0.11)

    # initialize random parameter
    t_random = random.uniform(0, 100)
    m_random = [random.uniform(0, 100), random.uniform(0, 100)]

    # initialize output
    image_strange_texture = np.zeros((oysize, oxsize, 3))

    # calculate color
    for y in nb.prange(oysize):
        for x in nb.prange(oxsize):

            # initial value
            value = int(x + t_random * 80.0 + m_random[0] * 10.0) ^ int(y + t_random * 80.0 + m_random[1] * 10.0)

            # update pixel value
            color = 1.0
            if value <= 1:
                color = background_value
            if value % 2 == 0 and value > 2:
                color = background_value
            for i in range(3, int(np.floor(np.sqrt(float(value)))), random.randint(1, 10)):
                if value % i == 0:
                    color = background_value

            # generate random color
            new_color = [
                color / random.uniform(0.01, 3),
                color / random.uniform(0.01, 3),
                color / random.uniform(0.01, 3),
            ]

            # update color
            image_strange_texture[y, x] = new_color

    # rotate texture randomly
    image_strange_texture = np.rot90(image_strange_texture, random.randint(0, 3))

    return image_strange_texture


def generate_stains_texture(oxsize, oysize):
    """Generate a stains similar texture using FFT.

    :param oxsize: The width of the output texture image.
    :type oxsize: int
    :param oysize: The height of the output texture image.
    :type oysize: int
    """

    # fixed internal resolution
    ysize, xsize = 200, 200

    wave_grid_output = np.zeros((ysize, xsize), dtype="uint8")

    for i in range(random.randint(3, 5)):
        # fixed resolution of the wave image
        resolution = random.uniform(0.1, 0.95)

        # Create a 2D grid of coordinates
        x_array = np.arange(-xsize / 2, xsize / 2) * resolution
        y_array = np.arange(-ysize / 2, ysize / 2) * resolution
        x_grid, y_grid = np.meshgrid(x_array, y_array)

        wave_grid_fft_shifted = np.zeros((ysize, xsize), dtype="complex")
        for i in range(random.randint(2, 3)):
            # iterations for adding waves
            iterations = random.randint(2, 4)
            wave_grid = np.zeros((ysize, xsize), dtype="float")
            for i in range(iterations):

                # Calculate the wave height using a sine function
                A = np.random.uniform(5, 15)  # Amplitude
                f = np.random.uniform(0.01, 0.05)  # Frequency
                p = np.random.uniform(0, 2 * np.pi)  # Phase
                kx = np.random.uniform(-1, 1)  # x-component of wave vector
                ky = np.random.uniform(-1, 1)  # y-component of wave vector
                h_sine = A * np.sin(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

                # Calculate the wave height using a cosine function
                A = np.random.uniform(5, 15)  # Amplitude
                f = np.random.uniform(0.01, 0.05)  # Frequency
                p = np.random.uniform(0, 2 * np.pi)  # Phase
                kx = np.random.uniform(-1, 1)  # x-component of wave vector
                ky = np.random.uniform(-1, 1)  # y-component of wave vector
                h_cosine = A * np.cos(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

                # combine heights from sine and cosine
                wave_grid = h_sine + h_cosine

            # Compute the FFT of the wave heights, shift the zero-frequency component to the center and then sum them
            wave_grid_fft_shifted += np.fft.fftshift(np.fft.fft2(wave_grid))

        # unshift the FFT component
        new_wave_grid = np.fft.ifft2(np.fft.ifftshift((wave_grid_fft_shifted)))

        # get the real part only
        new_wave_grid = np.real(new_wave_grid)

        # scale to 0 -1
        new_wave_grid = (new_wave_grid - new_wave_grid.min()) / (new_wave_grid.max() - new_wave_grid.min())

        # convert to uint8
        new_wave_grid = np.uint8(new_wave_grid * 255)

        # merge into output
        wave_grid_output += new_wave_grid

    # blur to smoothen texture
    wave_grid_output = cv2.GaussianBlur(wave_grid_output, (3, 3), 0)

    # remove low frequency area
    wave_grid_output = remove_frequency(wave_grid_output, random.randint(25, 35))

    # median filter to smoothen texture
    wave_grid_output = cv2.medianBlur(wave_grid_output, 3)

    # resize to output size
    wave_grid_output = cv2.resize(wave_grid_output, (oxsize, oysize), interpolation=cv2.INTER_LINEAR)

    return wave_grid_output


def generate_granular_texture(oxsize, oysize):
    """Generate random granular texture using FFT.

    :param oxsize: The width of the output texture image.
    :type oxsize: int
    :param oysize: The height of the output texture image.
    :type oysize: int
    """

    # fixed internal resolution
    ysize, xsize = 500, 500

    wave_grid_output = np.zeros((ysize, xsize), dtype="uint8")

    # fixed resolution of the wave image
    resolution = random.uniform(0.95, 0.95)

    # Create a 2D grid of coordinates
    x_array = np.arange(-xsize / 2, xsize / 2) * resolution
    y_array = np.arange(-ysize / 2, ysize / 2) * resolution
    x_grid, y_grid = np.meshgrid(x_array, y_array)

    # iterations for adding waves
    iterations = random.randint(1, 1)
    wave_grid = np.zeros((ysize, xsize), dtype="float")
    for i in range(iterations):

        # Calculate the wave height using a sine function
        A = np.random.uniform(5, 15)  # Amplitude
        f = np.random.uniform(0.01, 0.02)  # Frequency
        p = np.random.uniform(0, 2 * np.pi)  # Phase
        kx = np.random.uniform(-1, 1)  # x-component of wave vector
        ky = np.random.uniform(-1, 1)  # y-component of wave vector
        h_sine = A * np.sin(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

        # Calculate the wave height using a cosine function
        A = np.random.uniform(5, 15)  # Amplitude
        f = np.random.uniform(0.01, 0.02)  # Frequency
        p = np.random.uniform(0, 2 * np.pi)  # Phase
        kx = np.random.uniform(-1, 1)  # x-component of wave vector
        ky = np.random.uniform(-1, 1)  # y-component of wave vector
        h_cosine = A * np.cos(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

        # combine heights from sine and cosine
        wave_grid = h_sine + h_cosine

    # Compute the FFT of the wave heights, shift the zero-frequency component to the center and then sum them
    wave_grid_fft_shifted = np.fft.fftshift(np.fft.fft2(wave_grid))

    # unshift the FFT component
    new_wave_grid = np.fft.ifft2(np.fft.ifftshift((wave_grid_fft_shifted)))

    # get the real part only
    new_wave_grid = np.real(new_wave_grid)

    # scale to 0 -1
    new_wave_grid = (new_wave_grid - new_wave_grid.min()) / (new_wave_grid.max() - new_wave_grid.min())

    # convert to uint8
    wave_grid_output = np.uint8(new_wave_grid * 255)

    # blur to smoothen texture
    wave_grid_output = cv2.GaussianBlur(wave_grid_output, (3, 3), 0)

    # remove frequency > 10
    frequency = 10
    wave_grid_output = remove_frequency(wave_grid_output, frequency=frequency)

    # remove border textures
    offset = 50
    wave_grid_output = wave_grid_output[offset:-offset, offset:-offset]

    # rescale
    wave_grid_output = (wave_grid_output - wave_grid_output.min()) / (wave_grid_output.max() - wave_grid_output.min())
    wave_grid_output = 255 - np.uint8(wave_grid_output * 255)

    # remove frequency > 100
    frequency = 100
    wave_grid_output = remove_frequency(wave_grid_output, frequency)

    # rescale again
    wave_grid_output = (wave_grid_output - wave_grid_output.min()) / (wave_grid_output.max() - wave_grid_output.min())
    wave_grid_output = np.uint8(wave_grid_output * 255)

    # resize to output size
    wave_grid_output = cv2.resize(wave_grid_output, (oxsize, oysize), interpolation=cv2.INTER_LINEAR)

    return wave_grid_output


def generate_curvy_edge_texture(oxsize, oysize):
    """Generate a masked of curves based edge texture using FFT.

    :param oxsize: The width of the output texture image.
    :type oxsize: int
    :param oysize: The height of the output texture image.
    :type oysize: int
    """

    # fixed internal resolution
    ysize, xsize = 500, 500

    wave_grid_output = np.zeros((ysize, xsize), dtype="uint8")

    for i in range(random.randint(1, 1)):
        # fixed resolution of the wave image
        resolution = random.uniform(0.9, 0.95)

        # Create a 2D grid of coordinates
        x_array = np.arange(-xsize / 2, xsize / 2) * resolution
        y_array = np.arange(-ysize / 2, ysize / 2) * resolution
        x_grid, y_grid = np.meshgrid(x_array, y_array)

        wave_grid_fft_shifted = np.zeros((ysize, xsize), dtype="complex")
        for i in range(random.randint(1, 1)):
            # iterations for adding waves
            iterations = random.randint(3, 5)
            wave_grid = np.zeros((ysize, xsize), dtype="float")
            for i in range(iterations):

                # Calculate the wave height using a sine function
                A = np.random.uniform(5, 15)  # Amplitude
                f = np.random.uniform(0.05, 0.1)  # Frequency
                p = np.random.uniform(0, 2 * np.pi)  # Phase
                kx = np.random.uniform(-1, 1)  # x-component of wave vector
                ky = np.random.uniform(-1, 1)  # y-component of wave vector
                h_sine = A * np.sin(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

                # Calculate the wave height using a cosine function
                A = np.random.uniform(5, 15)  # Amplitude
                f = np.random.uniform(0.05, 0.1)  # Frequency
                p = np.random.uniform(0, 2 * np.pi)  # Phase
                kx = np.random.uniform(-1, 1)  # x-component of wave vector
                ky = np.random.uniform(-1, 1)  # y-component of wave vector
                h_cosine = A * np.cos(2 * np.pi * (f * (kx * x_grid + ky * y_grid) - p))

                # combine heights from sine and cosine
                wave_grid += h_sine + h_cosine

            # Compute the FFT of the wave heights, shift the zero-frequency component to the center and then sum them
            wave_grid_fft_shifted += np.fft.fftshift(np.fft.fft2(wave_grid))

        # unshift the FFT component
        new_wave_grid = np.fft.ifft2(np.fft.ifftshift((wave_grid_fft_shifted)))

        # get the real part only
        new_wave_grid = np.real(new_wave_grid)

        # scale to 0 -1
        new_wave_grid = (new_wave_grid - new_wave_grid.min()) / (new_wave_grid.max() - new_wave_grid.min())

        # convert to uint8
        new_wave_grid = np.uint8(new_wave_grid * 255)

        # merge into output
        wave_grid_output += new_wave_grid

    # blur to smoothen texture
    wave_grid_output = cv2.GaussianBlur(wave_grid_output, (3, 3), 0)

    # remove frequency > 100
    frequency = 100
    wave_grid_output = remove_frequency(wave_grid_output, frequency=frequency)

    # rescale
    wave_grid_output = (wave_grid_output - wave_grid_output.min()) / (wave_grid_output.max() - wave_grid_output.min())
    wave_grid_output = np.uint8(wave_grid_output * 255)

    # blur to smoothen trexture
    wave_grid_output = cv2.GaussianBlur(wave_grid_output, (9, 9), 0)

    # resize to output size
    wave_grid_output = cv2.resize(wave_grid_output, (oxsize, oysize), interpolation=cv2.INTER_LINEAR)

    return wave_grid_output


def remove_frequency(wave_grid_output, frequency):
    """Remove image area bigger than the input frequency by using FFT.

    :param wave_grid_output: The input image.
    :type wave_grid_output: numpy array
    :param frequency: The frequency threshold.
    :type frequency: int
    """

    ysize, xsize = wave_grid_output.shape[:2]

    cy, cx = ysize // 2, xsize // 2
    mask = np.ones((ysize, xsize), np.uint8)

    r = random.randint(frequency, frequency)

    # compute mask to remove low frequency area
    y, x = np.ogrid[:ysize, :xsize]
    mask_area = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    mask[mask_area] = 0

    # convert to fft and shift to zero-frequency
    wave_grid_output_fft = np.fft.fft2(wave_grid_output)
    wave_grid_output_fft_shifted = np.fft.fftshift(wave_grid_output_fft)

    # apply mask and inverse DFT
    wave_grid_output_fft_shifted *= mask
    wave_grid_output2_fft = np.fft.ifft2(np.fft.ifftshift(wave_grid_output_fft_shifted))
    wave_grid_output2 = np.abs(wave_grid_output2_fft)

    # normalize image back to 0 - 255
    wave_grid_output2 = (wave_grid_output2 - wave_grid_output2.min()) / (
        wave_grid_output2.max() - wave_grid_output2.min()
    )
    wave_grid_output = 255 - np.uint8(wave_grid_output2 * 255)

    return wave_grid_output


def rotate_image(mat, angle, white_background=1):
    """Rotates an image (angle in degrees) and expands image to avoid
    cropping.
    """

    if white_background:
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

    if white_background:
        rotated_mat = cv2.bitwise_not(rotated_mat)

    return rotated_mat


def rotate_image_PIL(image, angle, background_value=(0, 0, 0), expand=0):
    """Rotates an image (angle in degrees) by converting them to PIL image first."""

    # for single channel
    if len(image.shape) < 3 and isinstance(background_value, tuple):
        background_value = int(np.mean(background_value))
    image_PIL = Image.fromarray(image)
    rotated_image_PIL = image_PIL.rotate(angle, expand=expand, fillcolor=background_value)

    rotated_image = np.array(rotated_image_PIL)

    return rotated_image


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
@jit(nopython=True, cache=True, parallel=True)
def add_folding_noise(img, side, p=0.1):
    # side = flag to put more noise at certain side
    #   1  = left side
    #   0  = right side

    # get image dimension
    ysize, xsize = img.shape[:2]
    for y in nb.prange(ysize):
        for x in nb.prange(xsize):
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
def warp_fold(
    img,
    ysize,
    fold_noise,
    fold_x,
    fold_width_one_side,
    fold_y_shift,
    side,
    backdrop_color,
    fmask=0,
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

    if side == "left":
        # after distortion
        dtop_left = [xs, ys]
        dtop_right = [xe, ys + fold_y_shift]
        dbottom_left = [xs, ye]
        dbottom_right = [xe, ye + fold_y_shift]

        # image cropping points
        cxs = fold_x - fold_width_one_side
        cxe = fold_x
        cys = 0
        cye = ysize

    else:
        # after distortion
        dtop_left = [xs, ys + (fold_y_shift)]
        dtop_right = [xe, ys]
        dbottom_left = [xs, ye + (fold_y_shift)]
        dbottom_right = [xe, ye]

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
    cysize, cxsize = img_crop.shape[:2]
    if len(img_crop.shape) > 2:
        cdim = img_crop.shape[2]
    else:
        cdim = 2

    # darken the folded area
    # no darken effect for mask
    if not fmask:
        darken_ratio = random.uniform(0.99, 1.0)
        if len(img_crop.shape) > 2:
            # skip alpha layer, no darken for alpha layer
            for i in range(3):
                img_crop[:, :, i] = img_crop[:, :, i] * darken_ratio
        else:
            img_crop * darken_ratio

    # warp folding area
    img_warped = four_point_transform(
        img_crop,
        source_pts,
        destination_pts,
        cxsize,
        cysize + fold_y_shift,
    ).astype("uint8")

    # mask of warping
    img_mask = np.full_like(img_crop, fill_value=255, dtype="uint8")
    img_mask_warped = four_point_transform(
        img_mask,
        source_pts,
        destination_pts,
        cxsize,
        cysize + fold_y_shift,
    ).astype("uint8")

    # update color
    if cdim < 3:
        backdrop_color = np.mean(backdrop_color)
        img_warped[img_mask_warped < 255] = backdrop_color
    else:
        for i in range(3):
            img_warped[:, :, i][img_mask_warped[:, :, i] < 255] = backdrop_color[i]

    if fold_noise != 0:
        if side == "left":
            noise_side = 1
        else:
            noise_side = 0

        has_alpha = 0
        if cdim == 4:
            has_alpha = 1
            img_alpha = img_warped[:, :, 3]
            img_warped = img_warped[:, :, :3]

        img_warped = add_folding_noise(img_warped, noise_side, fold_noise / 2)

        if has_alpha:
            img_warped = np.dstack((img_warped, img_alpha))

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


@jit(nopython=True, cache=True, parallel=True)
def smooth(points, iterations):
    """
    Smooth points using chaikin method.

    :param points: a list of more than 2 points, where each point is a tuple/array of len=2
    :type points: array
    :param iter: number of times to apply chaikin algorithm
    :type iter: int
    :return:
    """

    percent = 0.25
    # iterations can't be parallelized due to we need the prior points
    for _ in range(iterations):
        current_ysize = points.shape[0]
        path = np.zeros((current_ysize * 2, 2), dtype="float")
        # first and last point are the same
        path[0] = points[0]
        path[-1] = points[-1]
        for i in nb.prange(current_ysize - 1):
            p0 = points[i]
            p1 = points[i + 1]
            # distance between x values of two subsequent points
            dx = p1[0] - p0[0]
            # distance between y values of two subsequent points
            dy = p1[1] - p0[1]
            # creating two new points having 25% and 75% distance from the previous point
            new_px0, new_py0 = p0[0] + dx * percent, p0[1] + dy * percent
            new_px1, new_py1 = p0[0] + dx * (1 - percent), p0[1] + dy * (1 - percent)

            # 2 new points per current single point
            n = ((i + 1) * 2) - 1
            path[n][0] = new_px0
            path[n][1] = new_py0
            path[n + 1][0] = new_px1
            path[n + 1][1] = new_py1

        # update points for next iteration
        points = path

    return points


def add_noise(image, intensity_range=(0.1, 0.2), color_range=(0, 224), noise_condition=0, image2=None):
    """Applies random noise to the input image.

    :param image: The image to noise.
    :type image: numpy.array
    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    :param noise_condition: Condition to apply noise mask.
    :type noise_condition: int, optional
    :param image2: Image for the noise evaluation.
    :type image2: numpy.array, optional
    """

    # check not None and use image2 as checking image
    if image2 is not None:
        checking_image = image2
    else:
        checking_image = image

    random_value = np.random.uniform(0, 1, size=image.shape)
    random_value2 = np.random.randint(color_range[0], color_range[1] + 1, size=image.shape)

    intensity = random.uniform(intensity_range[0], intensity_range[1])

    # find indices where sobelized image value == 255 and random value < intensity
    if noise_condition == 0:
        condition_evaluation = checking_image == 0
    elif noise_condition == 1:
        condition_evaluation = checking_image == 255
    elif noise_condition == 2:
        condition_evaluation = checking_image > 255
    elif noise_condition == 4:
        condition_evaluation = checking_image < 255

    condition_evaluation2 = random_value < intensity
    indices = np.logical_and(condition_evaluation, condition_evaluation2)

    # output
    image_noise = image.copy()

    # apply noise with indices
    image_noise[indices] = random_value2[indices]

    return image_noise


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
        # get input arguments for threshold function
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

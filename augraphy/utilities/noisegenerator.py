import random

import cv2
import numpy as np
from numba import config
from numba import jit
from sklearn.datasets import make_blobs


class NoiseGenerator:
    """Core object to generate mask of noise.

    :param noise_type: Types of generated noise.
        1 = default, even spread of noise
        2 = noise with regular pattern
        3 = noise at all borders of image
        4 = sparse and little noise
        5 = gaussian noise
        6 = perlin noise
    :type noise_type: int, optional
    :param noise_side: Location of generated noise. Choose from:
        "left", "right", "top", "bottom","top_left", "top_right", "bottom_left", "bottom_right".
    :type noise_side: string, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    """

    def __init__(self, noise_type=1, noise_side=None, numba_jit=1):
        self.noise_type = noise_type
        self.noise_side = noise_side
        self.numba_jit = numba_jit
        self.sides = [
            "left",
            "right",
            "top",
            "bottom",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]
        config.DISABLE_JIT = bool(1 - numba_jit)

    def compute_perlin(self, x, y, permutation_table):
        """Calculates Perlin noise values for each point on the grid using linear interpolation and gradient selection.

        :param x: Grid x coordinates of the Perlin noise values.
        :type x: numpy array
        :param y: Grid y coordinates of the Perlin noise values.
        :type y: numpy array
        :param permutation_table: Permutation table to shuffe and select the gradient vectors during the Perlin noise generation process.
        :type permutation_table: numpy array
        """

        xi, yi = x.astype(int), y.astype(int)
        xg, yg = x - xi, y - yi
        xf, yf = self.compute_fade(xg), self.compute_fade(yg)

        p00 = permutation_table[permutation_table[xi] + yi]
        p01 = permutation_table[permutation_table[xi] + yi + 1]
        p10 = permutation_table[permutation_table[xi + 1] + yi]
        p11 = permutation_table[permutation_table[xi + 1] + yi + 1]

        n00 = self.compute_gradient(p00, xg, yg)
        n01 = self.compute_gradient(p01, xg, yg - 1)
        n10 = self.compute_gradient(p10, xg - 1, yg)
        n11 = self.compute_gradient(p11, xg - 1, yg - 1)

        x1 = self.compute_lerp(n00, n10, xf)
        x2 = self.compute_lerp(n01, n11, xf)
        return self.compute_lerp(x1, x2, yf)

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_lerp(a, b, x):
        """Performs linear interpolation between two values based on a interpolation parameter.

        :param a: Starting value of the linear interpolation process.
        :type a: numpy array
        :param b: End value of the linear interpolation process.
        :type b: numpy array
        :param x: The weight of the interpolation process.
        :type x: numpy array
        """

        return a + x * (b - a)

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_fade(f):
        """Computes fade values for interpolation purpose.

        :param f: Input fractional value for interpolation purpose.
        :type f: numpy array
        """

        return 6 * f**5 - 15 * f**4 + 10 * f**3

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_gradient(c, x, y):
        """Computes the gradient vector calculates the dot product.

        :param c: Input array that determines the selection of the gradient vector.
        :type c: numpy array
        :param x: The x coordinates of the points on the grid
        :type x: numpy array
        :param y: The y coordinates of the points on the grid
        :type y: numpy array
        """

        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        rows, cols = c.shape

        result = np.empty_like(x)

        for i in range(rows):
            for j in range(cols):
                c_remainder = c[i, j] % 4
                gradient_co = vectors[c_remainder]
                result[i, j] = gradient_co[0] * x[i, j] + gradient_co[1] * y[i, j]

        return result

    def generate_perlin_noise(self, width, height, points=(5, 20)):
        """Generates Perlin noise for a grid of specified width and height.

        :param width: Width of the generated 2d perlin noise grid.
        :type width: int
        :param height: Height of the generated 2d perlin noise grid.
        :type height: int
        :param points: Number of points in each x and y direction within the grid.
        :type points: int
        """

        lin_array_x = np.linspace(0, np.random.randint(points[0], points[1]), width, endpoint=False)
        lin_array_y = np.linspace(0, np.random.randint(points[0], points[1]), height, endpoint=False)
        x, y = np.meshgrid(lin_array_x, lin_array_y)

        permutation_table = np.arange(256, dtype=int)
        np.random.shuffle(permutation_table)
        permutation_table = np.stack([permutation_table, permutation_table]).flatten()

        img_perlin = self.compute_perlin(x, y, permutation_table)
        return img_perlin

    def generate_clusters_and_samples(self, noise_type, noise_concentration, max_size):
        """Generate number of noise clusters and number of samples in each noise cluster.

        :param noise_type: Types of generated noise.
        :type noise_type: int
        :param noise_concentration: Pair of floats determining concentration of noise.
        :type noise_concentration: tuple
        :param max_size: Maximum between width and height of image.
        :type max_size: int
        """

        if noise_type == 4:
            n_clusters = (
                int((noise_concentration[0]) * (max_size / 50)),
                int((noise_concentration[1]) * (max_size / 30)),
            )
            n_samples = n_clusters
        # default: following input value
        else:
            n_clusters = (
                int((noise_concentration[0]) * max_size),
                int((noise_concentration[1]) * max_size),
            )
            n_samples = (
                int((noise_concentration[0]) * max_size),
                int((noise_concentration[1]) * max_size),
            )

        # prevent 0 cluster or 0 sample
        n_clusters = (max(1, n_clusters[0]), max(1, n_clusters[1]))
        n_samples = (max(10, n_samples[0]), max(10, n_samples[1]))

        # generate array of samples
        n_samples_array = [
            random.randint(n_samples[0], n_samples[1]) for _ in range(random.randint(n_clusters[0], n_clusters[1]))
        ]

        return n_samples_array

    def generate_sparsity_std(self, noise_type, noise_side, noise_sparsity, xsize, ysize, max_size):
        """Generate standard deviation(std) to control the sparsity of the noise.

        :param noise_type: Types of generated noise.
        :type noise_type: int
        :param noise_side: Location of generated noise.
        :type noise_side: string
        :param noise_sparsity: Pair of floats determining sparseness of noise.
        :type noise_sparsity: tuple
        :param xsize: Width of image.
        :type xsize: int
        :param ysize: Height of image.
        :type ysize: int
        :param max_size: Maximum between width and height of image.
        :type max_size: int
        """

        if noise_type == 2:
            std_range = (
                int((noise_sparsity[0]) * (max_size / 5)),
                int((noise_sparsity[1]) * (max_size / 5)),
            )
        elif noise_type == 3:
            std_range = (
                int((noise_sparsity[0]) * (max_size / 3)),
                int((noise_sparsity[1]) * (max_size / 3)),
            )
        else:
            std_range = (
                int((noise_sparsity[0]) * (max_size)),
                int((noise_sparsity[1]) * (max_size)),
            )

        # for noise concentrated at certain part of image
        # left
        if noise_side == "left":
            center_x = (0, 0)
            center_y = (0, ysize)
        # right
        elif noise_side == "right":
            center_x = (xsize, xsize)
            center_y = (0, ysize)
        # top
        elif noise_side == "top":
            center_x = (0, xsize)
            center_y = (0, 0)
        # bottom
        elif noise_side == "bottom":
            center_x = (0, xsize)
            center_y = (ysize, ysize)
        # top_left
        elif noise_side == "top_left":
            center_x = (0, int(xsize / 4))
            center_y = (0, int(ysize / 4))
        # top_right
        elif noise_side == "top_right":
            center_x = (int(xsize * 0.75), xsize)
            center_y = (0, int(ysize / 4))
        # bottom_left
        elif noise_side == "bottom_left":
            center_x = (0, int(xsize / 4))
            center_y = (int(ysize * 0.75), ysize)
        # bottom_right
        elif noise_side == "bottom_right":
            center_x = (int(xsize * 0.75), xsize)
            center_y = (int(ysize * 0.75), ysize)
        else:
            center_x = (0, xsize)
            center_y = (0, ysize)

        std = random.randint(std_range[0], std_range[1])

        return std, center_x, center_y

    def generate_points(self, n_samples_array, std, center_x, center_y, xsize, ysize):
        """Generate x&y coordinates of noise.

        :param n_samples_array: List contains number of points sample for each cluster.
        :type n_samples_array: list
        :param std: Standard deviation to determine sparseness of generated points.
        :type std: int
        :param center_x: Center x of the generated noises.
        :type center_x: int
        :param center_y: Center y of the generated noises.
        :type center_y: int
        :param xsize: Width of image.
        :type xsize: int
        :param ysize: Height of image.
        :type ysize: int
        """

        # generate clusters of blobs
        generated_points_x, _ = make_blobs(
            n_samples=n_samples_array,
            center_box=center_x,
            cluster_std=std,
            n_features=1,
        )

        generated_points_y, _ = make_blobs(
            n_samples=n_samples_array,
            center_box=center_y,
            cluster_std=std,
            n_features=1,
        )

        # remove decimals
        generated_points_x = generated_points_x.astype("int")
        generated_points_y = generated_points_y.astype("int")

        ind_delete = np.logical_or.reduce(
            (
                generated_points_x < 0,
                generated_points_y < 0,
                generated_points_x > xsize - 1,
                generated_points_y > ysize - 1,
            ),
        )
        generated_points_y = np.delete(generated_points_y, ind_delete.reshape(ind_delete.shape[0]), axis=0)
        generated_points_x = np.delete(generated_points_x, ind_delete.reshape(ind_delete.shape[0]), axis=0)

        return generated_points_x, generated_points_y

    def generate_mask(
        self,
        noise_background,
        noise_value,
        generated_points_x,
        generated_points_y,
        xsize,
        ysize,
    ):
        """Generate mask of noise.

        :param noise_background: Tuple of ints to determine background value of mask.
        :type noise_background: tuple
        :param noise_value: Tuple of ints to determine value of noise.
        :type noise_value: tuple
        :param generated_points_x: x point value of noise.
        :type generated_points_x: numpy array
        :param generated_points_y: y point value of noise.
        :type generated_points_y: numpy array
        :param xsize: Width of image.
        :type xsize: int
        :param ysize: Height of image.
        :type ysize: int
        """

        # background of noise mask
        img_mask = np.random.randint(
            noise_background[0],
            noise_background[1] + 1,
            (ysize, xsize),
        )

        # mask of random value
        img_mask_random = np.random.randint(
            low=noise_value[0],
            high=noise_value[1] + 1,
            size=(ysize, xsize),
        )

        # insert random value into background
        img_mask[generated_points_y, generated_points_x] = img_mask_random[generated_points_y, generated_points_x]

        return img_mask.astype("uint8")

    def generate_mask_main(
        self,
        noise_type,
        noise_side,
        noise_value,
        noise_background,
        noise_sparsity,
        noise_concentration,
        xsize,
        ysize,
    ):
        """Main function to generate mask of noise in each iteration.

        :param noise_type: Types of generated noise.
        :type noise_type: int
        :param noise_side: Location of generated noise.
        :type noise_side: string
        :param noise_value: Tuple of ints to determine value of noise.
        :type noise_value: tuple
        :param noise_background: Tuple of ints to determine background value of mask.
        :type noise_background: tuple
        :param noise_sparsity: Pair of floats determining sparseness of noise.
        :type noise_sparsity: tuple, optional
        :param noise_concentration: Pair of floats determining concentration of noise.
        :type noise_concentration: tuple, optional
        :param xsize: Width of image.
        :type xsize: int
        :param ysize: Height of image.
        :type ysize: int
        """

        if noise_type not in [1, 2, 3, 4]:

            # gaussian noise
            if noise_type == 5:
                img_mask = np.full((ysize, xsize), fill_value=255, dtype="float")

                iterations = random.randint(3, 8)
                for _ in range(iterations):
                    # random parameters for gaussian noise
                    mean = random.randint(0, 255)
                    sigma = random.randint(0, 255)
                    ratio = random.randint(5, 10)

                    # generate random gaussian noise
                    img_gaussian = np.random.normal(mean, sigma, (int(xsize / ratio), int(ysize / ratio)))
                    img_gaussian = cv2.resize(img_gaussian, (xsize, ysize), interpolation=cv2.INTER_LINEAR)

                    # add gaussian noise
                    img_mask += img_gaussian

                # change image to uint8 after the summation
                img_mask = img_mask.astype("uint8")

            # perlin noise
            elif noise_type == 6:

                img_mask = self.generate_perlin_noise(xsize, ysize)
                img_mask = (img_mask - np.min(img_mask)) / (np.max(img_mask) - np.min(img_mask))
                img_mask = (img_mask * 255).astype("uint8")

            # threshold to increase or decrease the noise concentration
            noise_threshold = int(random.uniform(noise_concentration[0], noise_concentration[1]) * 255)

            # mask of background
            img_background = np.random.randint(
                noise_background[0],
                noise_background[1] + 1,
                size=(ysize, xsize),
                dtype="uint8",
            )
            indices_background = img_mask > noise_threshold

            # temporary assignment
            min_value = np.min(img_mask)
            img_mask[indices_background] = min_value

            # scale value to provided input range
            current_min = np.min(img_mask)
            current_max = np.max(img_mask)
            img_mask = (
                ((img_mask.astype("float") - current_min) / (current_max - current_min))
                * (noise_value[1] - noise_value[0])
                + noise_value[0]
            ).astype("uint8")

            # update background value
            img_mask[indices_background] = img_background[indices_background]

            # generate sparsity
            sparsity = random.uniform(noise_sparsity[0], noise_sparsity[1])

            # reduce noise area based on sparsity value
            img_sparsity_value = np.arange(-0.3, 1, 1 / int(ysize * sparsity), dtype="float")
            length = len(img_sparsity_value)
            img_sparsity_value = img_sparsity_value.reshape(length, 1).repeat(xsize, 1)
            img_sparsity_value[img_sparsity_value < 0] = 0
            img_sparsity_value = cv2.resize(
                img_sparsity_value,
                (xsize, int(ysize * sparsity)),
                interpolation=cv2.INTER_LINEAR,
            )
            img_sparsity_value *= 255
            img_sparsity_mask = np.full((ysize, xsize), fill_value=255, dtype="float")

            # map noise image back to mask based on noise side
            if noise_side == "top":
                img_sparsity_mask[: int(ysize * sparsity), :] = img_sparsity_value
            elif noise_side == "left":
                img_sparsity_value = cv2.resize(
                    np.rot90(img_sparsity_value, 1),
                    (int(xsize * sparsity), ysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_mask[:, : int(xsize * sparsity)] = img_sparsity_value
            elif noise_side == "right":
                img_sparsity_value = cv2.resize(
                    np.rot90(img_sparsity_value, 3),
                    (int(xsize * sparsity), ysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_mask[:, -int(xsize * sparsity) :] = img_sparsity_value
            elif noise_side == "bottom":
                img_sparsity_value = np.flipud(img_sparsity_value)
                img_sparsity_mask[-int(ysize * sparsity) :, :] = img_sparsity_value
            elif noise_side == "top_left":
                cysize, cxsize = img_sparsity_value.shape[:2]
                img_sparsity_value_rot = cv2.resize(
                    np.rot90(img_sparsity_value, 3),
                    (cxsize, cysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_value = img_sparsity_value * img_sparsity_value_rot
                img_sparsity_value = (img_sparsity_value - np.min(img_sparsity_value)) / (
                    np.max(img_sparsity_value) - np.min(img_sparsity_value)
                )
                img_sparsity_value = 255 - (img_sparsity_value * 255)
                img_sparsity_mask[: int(ysize * sparsity), :] = np.flipud(img_sparsity_value)
            elif noise_side == "top_right":
                cysize, cxsize = img_sparsity_value.shape[:2]
                img_sparsity_value_rot = cv2.resize(
                    np.rot90(img_sparsity_value, 1),
                    (cxsize, cysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_value = img_sparsity_value * img_sparsity_value_rot
                img_sparsity_value = (img_sparsity_value - np.min(img_sparsity_value)) / (
                    np.max(img_sparsity_value) - np.min(img_sparsity_value)
                )
                img_sparsity_value = 255 - (img_sparsity_value * 255)
                img_sparsity_mask[: int(ysize * sparsity), :] = np.flipud(img_sparsity_value)
            elif noise_side == "bottom_left":
                cysize, cxsize = img_sparsity_value.shape[:2]
                img_sparsity_value_rot = cv2.resize(
                    np.rot90(img_sparsity_value, 3),
                    (cxsize, cysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_value = img_sparsity_value * img_sparsity_value_rot
                img_sparsity_value = (img_sparsity_value - np.min(img_sparsity_value)) / (
                    np.max(img_sparsity_value) - np.min(img_sparsity_value)
                )
                img_sparsity_value = 255 - (img_sparsity_value * 255)
                img_sparsity_mask[-int(ysize * sparsity) :, :] = img_sparsity_value
            elif noise_side == "bottom_right":
                cysize, cxsize = img_sparsity_value.shape[:2]
                img_sparsity_value_rot = cv2.resize(
                    np.rot90(img_sparsity_value, 1),
                    (cxsize, cysize),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_sparsity_value = img_sparsity_value * img_sparsity_value_rot
                img_sparsity_value = (img_sparsity_value - np.min(img_sparsity_value)) / (
                    np.max(img_sparsity_value) - np.min(img_sparsity_value)
                )
                img_sparsity_value = 255 - (img_sparsity_value * 255)
                img_sparsity_mask[-int(ysize * sparsity) :, :] = img_sparsity_value

            # multiply noise with sparsity mask
            img_mask = 255 - cv2.multiply(255 - img_mask, 255 - img_sparsity_mask.astype("uint8"), scale=1 / 255)

        # noise type 1, 2, 3, 4 - generate noise with scikit-learn's make_blobs
        else:
            # get max of y or x size
            max_size = max(xsize, ysize)

            # generate number of clusters and number of samples in each cluster
            n_samples_array = self.generate_clusters_and_samples(
                noise_type,
                noise_concentration,
                max_size,
            )

            # For sparsity of the noises (distance to centroid of cluster)
            std, center_x, center_y = self.generate_sparsity_std(
                noise_type,
                noise_side,
                noise_sparsity,
                xsize,
                ysize,
                max_size,
            )

            if noise_type == 2:

                # reduce sparsity
                std = int(std / 5)

                # size of noise depends on noise sparsity
                random_sparsity = np.random.uniform(noise_sparsity[0], noise_sparsity[1])
                end_y = max((random_sparsity) * ysize, int(ysize / 10))

                # randomize noise pattern
                n_step_x = int(xsize / random.randint(10, 14))
                n_step_y = int(ysize / random.randint(16, 20))

                # initialize points array
                generated_points_x = np.array([[-1]], dtype="int")
                generated_points_y = np.array([[-1]], dtype="int")

                # initial noise location
                ccenter_y = (0, 0)
                ccenter_x = (0, 0)

                while ccenter_y[1] < end_y:

                    # reduce sample to generate gradient in noise
                    samples_index = np.ceil(len(n_samples_array) / 2).astype("int")
                    n_samples_array = n_samples_array[:samples_index]

                    # varying y
                    ccenter_y = (ccenter_y[1], ccenter_y[1] + n_step_y)
                    ccenter_x = (0, 0)

                    check_break = 0
                    while True:
                        # varying x
                        ccenter_x = (ccenter_x[0], ccenter_x[1] + n_step_x)

                        # generate coordinates for clusters of blobs
                        cgenerated_points_x, cgenerated_points_y = self.generate_points(
                            n_samples_array,
                            int(std / 5),
                            ccenter_x,
                            ccenter_y,
                            xsize,
                            ysize,
                        )

                        # combine coordinates
                        generated_points_x = np.concatenate(
                            [generated_points_x, cgenerated_points_x],
                        )
                        generated_points_y = np.concatenate(
                            [generated_points_y, cgenerated_points_y],
                        )

                        # space between next noise patch
                        add_space = random.randint(10, 20)
                        ccenter_x = (
                            ccenter_x[0] + n_step_x + add_space,
                            ccenter_x[1] + n_step_x + add_space,
                        )

                        # to break out from inner loop
                        if check_break:
                            break
                        elif ccenter_x[1] > xsize:
                            ccenter_x = (xsize - 1, xsize - 1)
                            check_break = 1

                    # space between next noise patch
                    add_space = random.randint(5, 15)

                    ccenter_y = (ccenter_y[1] + add_space, ccenter_y[1] + add_space)

                # generate mask
                img_mask = self.generate_mask(
                    noise_background,
                    noise_value,
                    generated_points_x,
                    generated_points_y,
                    xsize,
                    ysize,
                )

                # rotate mask according to noise_side
                if noise_side == "top" or noise_side == "top_left" or noise_side == "top_right":
                    img_mask = img_mask
                elif noise_side == "bottom" or noise_side == "bottom_left" or noise_side == "bottom_right":
                    img_mask = np.flipud(img_mask)
                elif noise_side == "left":
                    img_mask = np.rot90(img_mask, 1)
                elif noise_side == "right":
                    img_mask = np.rot90(img_mask, 3)
                else:
                    img_mask = np.rot90(img_mask, random.randint(0, 3))

            else:
                # generate coordinates for clusters of blobs
                generated_points_x, generated_points_y = self.generate_points(
                    n_samples_array,
                    std,
                    center_x,
                    center_y,
                    xsize,
                    ysize,
                )

                # generate mask
                img_mask = self.generate_mask(
                    noise_background,
                    noise_value,
                    generated_points_x,
                    generated_points_y,
                    xsize,
                    ysize,
                )

                # rotate and merge mask into 4 sides
                if noise_type == 3:
                    img_mask = np.minimum(
                        img_mask,
                        cv2.resize(np.rot90(img_mask), (xsize, ysize), interpolation=cv2.INTER_AREA),
                    )
                    img_mask = np.minimum(
                        img_mask,
                        cv2.resize(np.rot90(img_mask, k=2), (xsize, ysize), interpolation=cv2.INTER_AREA),
                    )

        return img_mask

    def generate_noise(
        self,
        noise_iteration=(1, 1),
        noise_size=(1, 1),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.4, 0.6),
        noise_concentration=(0.4, 0.6),
        xsize=1500,
        ysize=1500,
    ):
        """Main function to generate noise.

        :param noise_iteration: Pair of ints to determine number of iterations to apply noise in the mask.
        :type noise_type: tuple, optional
        :param noise_size: Pair of ints to determine scale of noise in the mask.
        :type noise_size: tuple, optional
        :param noise_value: Pair of ints determining value of noise.
        :type noise_value: tuple, optional
        :param noise_background: Pair of ints determining value of noise background.
        :type noise_background: tuple, optional
        :param noise_sparsity: Pair of floats determining sparseness of noise.
        :type noise_sparsity: tuple, optional
        :param noise_concentration: Pair of floats determining concentration of noise.
        :type noise_concentration: tuple, optional
        :param xsize: Number of columns in generated mask of noise.
        :type xsize: int, optional
        :param ysize: Number of rows in generated mask of noise.
        :type ysize: int, optional
        """

        # generate random iterations
        iterations = random.randint(noise_iteration[0], noise_iteration[1])

        # generate background value
        background_value = random.randint(noise_background[0], noise_background[1])

        # initialize blank noise mask
        img_mask = np.full((ysize, xsize), fill_value=background_value, dtype="int")

        # any invalid noise type will reset noise type to 0
        if self.noise_type not in [1, 2, 3, 4, 5, 6]:
            noise_type = random.randint(1, 6)
        else:
            noise_type = self.noise_type

        # random location with no sides if no side is chosen
        if self.noise_side not in self.sides:
            noise_side = random.choice(self.sides)
        else:
            noise_side = self.noise_side

        # loop each iterations
        for _ in range(iterations):

            # divider to rescale noise mask to larger size
            y_divider = random.randint(noise_size[0], noise_size[1])
            x_divider = random.randint(noise_size[0], noise_size[1])

            # generate noise mask for current iteration
            img_mask_temporary = self.generate_mask_main(
                noise_type,
                noise_side,
                noise_value,
                noise_background,
                noise_sparsity,
                noise_concentration,
                int(xsize / x_divider),
                int(ysize / y_divider),
            )
            img_mask_temporary = cv2.resize(
                img_mask_temporary.astype("uint8"),
                (xsize, ysize),
                interpolation=cv2.INTER_CUBIC,
            )

            # merge noise mask in each iteration by getting their min value
            img_mask = np.minimum(img_mask_temporary, img_mask)

        # output needs uint8 type
        img_mask = img_mask.astype("uint8")

        return img_mask

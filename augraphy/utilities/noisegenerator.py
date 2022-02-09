import random

import cv2
import numpy as np
from sklearn.datasets import make_blobs


class NoiseGenerator:
    """Core object to generate mask of noise."""

    def __init__(self, noise_type=1, noise_side=None):
        self.noise_type = noise_type
        self.noise_side = noise_side
        self.sides = [
            "random",
            "left",
            "right",
            "top",
            "bottom",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]
        """
        noise_type:
        1 = default, even spread of noise
        2 = noise with regular pattern
        3 = noise at all borders of image
        4 = sparse and little noise

        noise_side:
        1. All sides valid for noise_type = 1 and 4
        2. For noise_type = 2, valid noise_side are only left, right, top and bottom.
        3. For noise_type = 3, noise_side is not applicable.

        """

        # any invalid noise type will reset noise type to 0
        if self.noise_type not in [1, 2, 3, 4]:
            self.noise_type = 1

        # random location with no sides if no side is chosen
        if self.noise_side not in self.sides:
            self.noise_side = "random"

    def generate_clusters_and_samples(self, noise_concentration, max_size):
        """
        Generate number of noise clusters and number of samples in each
        noise cluster.
        """

        if self.noise_type == 4:
            n_clusters = (
                int((noise_concentration[0]) * (max_size / 50)),
                int((noise_concentration[1]) * (max_size / 30)),
            )
            n_samples = n_clusters
        # default: following input value
        else:
            n_clusters = (int((noise_concentration[0]) * max_size), int((noise_concentration[1]) * max_size))
            n_samples = (int((noise_concentration[0]) * max_size), int((noise_concentration[1]) * max_size))

        # prevent 0 cluster or 0 sample
        n_clusters = (max(1, n_clusters[0]), max(1, n_clusters[1]))
        n_samples = (max(10, n_samples[0]), max(10, n_samples[1]))

        # generate array of samples
        n_samples_array = [
            random.randint(n_samples[0], n_samples[1]) for _ in range(random.randint(n_clusters[0], n_clusters[1]))
        ]

        return n_samples_array

    def generate_sparsity_std(self, noise_sparsity, xsize, ysize, max_size):
        """
        Generate standard deviation(std) to control the sparsity of the noise.
        """

        if self.noise_type == 2:
            std_range = (int((noise_sparsity[0]) * (max_size / 5)), int((noise_sparsity[1]) * (max_size / 5)))
        elif self.noise_type == 3:
            std_range = (int((noise_sparsity[0]) * (max_size / 3)), int((noise_sparsity[1]) * (max_size / 3)))
        else:
            std_range = (int((noise_sparsity[0]) * (max_size)), int((noise_sparsity[1]) * (max_size)))

        # for noise concentrated at certain part of image
        # left
        if self.noise_side == "left":
            center_x = (0, 0)
            center_y = (0, ysize)
        # right
        elif self.noise_side == "right":
            center_x = (xsize, xsize)
            center_y = (0, ysize)
        # top
        elif self.noise_side == "top":
            center_x = (0, xsize)
            center_y = (0, 0)
        # bottom
        elif self.noise_side == "bottom":
            center_x = (0, xsize)
            center_y = (ysize, ysize)
        # top_left
        elif self.noise_side == "top_left":
            center_x = (0, int(xsize / 4))
            center_y = (0, int(ysize / 4))
        # top_right
        elif self.noise_side == "top_right":
            center_x = (int(xsize * 0.75), xsize)
            center_y = (0, int(ysize / 4))
        # bottom_left
        elif self.noise_side == "bottom_left":
            center_x = (0, int(xsize / 4))
            center_y = (int(ysize * 0.75), ysize)
        # bottom_right
        elif self.noise_side == "bottom_right":
            center_x = (int(xsize * 0.75), xsize)
            center_y = (int(ysize * 0.75), ysize)
        else:
            center_x = (0, xsize)
            center_y = (0, ysize)

        std = random.randint(std_range[0], std_range[1])

        return std, center_x, center_y

    def generate_points(self, n_samples_array, std, center_x, center_y, xsize, ysize):
        """
        Generate x&y coordinates of noise.
        """
        # generate clusters of blobs
        if self.noise_type == 3:
            # left
            generated_points_x_left, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, 0),
                cluster_std=std,
                n_features=1,
            )
            generated_points_y_left, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, ysize),
                cluster_std=std,
                n_features=1,
            )
            # right
            generated_points_x_right, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(xsize, xsize),
                cluster_std=std,
                n_features=1,
            )
            generated_points_y_right, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, ysize),
                cluster_std=std,
                n_features=1,
            )
            # top
            generated_points_x_top, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, xsize),
                cluster_std=std,
                n_features=1,
            )
            generated_points_y_top, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, 0),
                cluster_std=std,
                n_features=1,
            )
            # bottom
            generated_points_x_bottom, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(0, xsize),
                cluster_std=std,
                n_features=1,
            )
            generated_points_y_bottom, _ = make_blobs(
                n_samples=n_samples_array,
                center_box=(ysize, ysize),
                cluster_std=std,
                n_features=1,
            )

            generated_points_x = np.concatenate(
                (generated_points_x_left, generated_points_x_right, generated_points_x_top, generated_points_x_bottom),
            )
            generated_points_y = np.concatenate(
                (generated_points_y_left, generated_points_y_right, generated_points_y_top, generated_points_y_bottom),
            )

        else:
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

        # delete invalid points (smaller or bigger than image size)
        ind_delete_x = np.where(generated_points_x < 0)
        generated_points_x = np.delete(generated_points_x, ind_delete_x, axis=0)
        generated_points_y = np.delete(generated_points_y, ind_delete_x, axis=0)

        ind_delete_y = np.where(generated_points_y < 0)
        generated_points_y = np.delete(generated_points_y, ind_delete_y, axis=0)
        generated_points_x = np.delete(generated_points_x, ind_delete_y, axis=0)

        ind_delete_x = np.where(generated_points_x > xsize - 1)
        generated_points_x = np.delete(generated_points_x, ind_delete_x, axis=0)
        generated_points_y = np.delete(generated_points_y, ind_delete_x, axis=0)

        ind_delete_y = np.where(generated_points_y > ysize - 1)
        generated_points_y = np.delete(generated_points_y, ind_delete_y, axis=0)
        generated_points_x = np.delete(generated_points_x, ind_delete_y, axis=0)

        return generated_points_x, generated_points_y

    def generate_mask(self, noise_background, noise_value, generated_points_x, generated_points_y, xsize, ysize):
        """
        Generate mask of noise.
        """

        # background of noise mask
        img_mask = np.random.randint(noise_background[0], noise_background[1] + 1, (ysize, xsize))

        # mask of random value
        img_mask_random = np.random.randint(low=noise_value[0], high=noise_value[1] + 1, size=(ysize, xsize))

        # get xy points in list form
        x_points = list(generated_points_x)
        y_points = list(generated_points_y)

        # insert random value into background
        img_mask[y_points, x_points] = img_mask_random[y_points, x_points]

        return img_mask

    def generate_mask_main(self, noise_value, noise_background, noise_sparsity, noise_concentration, xsize, ysize):
        """
        Main function to generate mask of noise in each iteration.
        """

        # get max of y or x size
        max_size = max(xsize, ysize)

        # generate number of clusters and number of samples in each cluster
        n_samples_array = self.generate_clusters_and_samples(noise_concentration, max_size)

        # For sparsity of the noises (distance to centroid of cluster)
        std, center_x, center_y = self.generate_sparsity_std(noise_sparsity, xsize, ysize, max_size)

        if self.noise_type == 2:

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
                ccenter_y = [ccenter_y[1], ccenter_y[1] + n_step_y]
                ccenter_x = (0, 0)

                check_break = 0
                while True:
                    # varying x
                    ccenter_x = [ccenter_x[0], ccenter_x[1] + n_step_x]

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
                    generated_points_x = np.concatenate([generated_points_x, cgenerated_points_x])
                    generated_points_y = np.concatenate([generated_points_y, cgenerated_points_y])

                    # space between next noise patch
                    add_space = random.randint(10, 20)
                    ccenter_x = [ccenter_x[0] + n_step_x + add_space, ccenter_x[1] + n_step_x + add_space]

                    # to break out from inner loop
                    if check_break:
                        break
                    elif ccenter_x[1] > xsize:
                        ccenter_x[0] = xsize - 1
                        ccenter_x[1] = xsize - 1
                        check_break = 1

                # space between next noise patch
                add_space = random.randint(5, 15)

                ccenter_y = [ccenter_y[1] + add_space, ccenter_y[1] + add_space]

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
            if self.noise_side == "top" or self.noise_side == "top_left" or self.noise_side == "top_right":
                img_mask = img_mask
            elif self.noise_side == "bottom" or self.noise_side == "bottom_left" or self.noise_side == "bottom_right":
                img_mask = np.flipud(img_mask)
            elif self.noise_side == "left":
                img_mask = np.rot90(img_mask, 1)
            elif self.noise_side == "right":
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
        """
        Main function to generate noise
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
        img_mask = np.full((xsize, ysize), fill_value=background_value).astype("int")

        # loop each iterations
        for _ in range(iterations):

            # divider to rescale noise mask to larger size
            y_divider = random.randint(noise_size[0], noise_size[1])
            x_divider = random.randint(noise_size[0], noise_size[1])

            # generate noise mask for current iteration
            img_mask_temporary = self.generate_mask_main(
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

            # merge noise mask in each iteration
            img_mask *= img_mask_temporary.astype("int")

            # rescale value from 0 - 255
            max_value = 255 if np.max(img_mask) > 255 else np.max(img_mask)
            img_mask = (img_mask - np.min(img_mask)) * max_value / (np.max(img_mask) - np.min(img_mask))
            img_mask[img_mask_temporary <= noise_value[1]] = img_mask_temporary[img_mask_temporary <= noise_value[1]]

        # output needs uint8 type
        img_mask = img_mask.astype("uint8")

        return img_mask

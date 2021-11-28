import random

import numpy as np
from sklearn.datasets import make_blobs


class NoiseGenerator:
    """
    Core object to generate mask of noise.
    """

    def __init__(self, noise_type=1):
        self.noise_type = noise_type
        """
        noise types:
        1 = single cluster noise
        2 = noise with regular pattern
        3 = noise at the borders of image
        4 = sparse and little noise
        5 = noise concentrated at left edge
        6 = noise concentrated at right edge
        7 = noise concentrated at top edge
        8 = noise concentrated at bottom edge
        """

    def generate_clusters_and_samples(self, noise_concentration, max_size):
        """
        Generate number of noise clusters and number of samples in each
        noise cluster.
        """
        if self.noise_type == 1:
            n_clusters = (1, 1)
            n_samples = (int((noise_concentration[0]) * max_size * 100), int((noise_concentration[1]) * max_size * 100))
        elif self.noise_type == 2:
            n_clusters = (
                int((noise_concentration[0]) * (max_size / 100)),
                int((noise_concentration[1]) * (max_size / 80)),
            )
            n_samples = (
                int((noise_concentration[0] ** 0.1) * (max_size * 20)),
                int((noise_concentration[1] ** 0.1) * (max_size * 30)),
            )
        elif self.noise_type == 4:
            n_clusters = (
                int((noise_concentration[0]) * (max_size / 50)),
                int((noise_concentration[1]) * (max_size / 30)),
            )
            n_samples = n_clusters
        # default: following input value
        else:
            n_clusters = (int((noise_concentration[0]) * max_size), int((noise_concentration[1]) * max_size))
            n_samples = (int((noise_concentration[0]) * max_size), int((noise_concentration[1]) * max_size))

        # generate array of samples
        n_samples_array = [
            random.randint(n_samples[0], n_samples[1]) for _ in range(random.randint(n_clusters[0], n_clusters[1]))
        ]

        return n_samples_array

    def generate_sparsity_std(self, noise_sparsity, xsize, ysize, max_size):
        """
        Generate standard deviation(std) to control the sparsity of the noise.
        """

        std_range = (int((noise_sparsity[0]) * (max_size)), int((noise_sparsity[1]) * (max_size)))
        center_x = (0, xsize)
        center_y = (0, ysize)

        if self.noise_type == 1:
            std_range = (int((noise_sparsity[0]) * (max_size / 10)), int((noise_sparsity[1]) * (max_size / 10)))
            center_x = (0, xsize)
            center_y = (0, ysize)
        if self.noise_type == 2:
            std_range = (int((noise_sparsity[0]) * (max_size / 10)), int((noise_sparsity[1]) * (max_size / 10)))
            center_x = (0, xsize)
            center_y = (0, ysize)
        elif (
            self.noise_type == 3
            or self.noise_type == 5
            or self.noise_type == 6
            or self.noise_type == 7
            or self.noise_type == 8
        ):

            std_range = (int((noise_sparsity[0]) * (max_size / 5)), int((noise_sparsity[1]) * (max_size / 5)))

            # left
            if self.noise_type == 5:
                center_x = (0, 0)
                center_y = (0, ysize)
            # right
            elif self.noise_type == 6:
                center_x = (xsize, xsize)
                center_y = (0, ysize)
            # top
            elif self.noise_type == 7:
                center_x = (0, xsize)
                center_y = (0, 0)
            # bottom
            elif self.noise_type == 8:
                center_x = (0, xsize)
                center_y = (ysize, ysize)

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

    def generate_noise(
        self,
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.4, 0.6),
        noise_concentration=(0.4, 0.6),
        xsize=1500,
        ysize=1500,
    ):
        """
        Main function to generate noise
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

        # get max of y or x size
        max_size = max(xsize, ysize)

        # generate number of clusters and number of samples in each cluster
        n_samples_array = self.generate_clusters_and_samples(noise_concentration, max_size)

        # For sparsity of the noises (distance to centroid of cluster)
        std, center_x, center_y = self.generate_sparsity_std(noise_sparsity, xsize, ysize, max_size)

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


# usage example
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from time import time

    start_time = time()

    # initialize noise generator
    noise_generator_type_1 = NoiseGenerator(noise_type=1)
    noise_generator_type_2 = NoiseGenerator(noise_type=2)
    noise_generator_type_3 = NoiseGenerator(noise_type=3)
    noise_generator_type_4 = NoiseGenerator(noise_type=4)
    noise_generator_type_5 = NoiseGenerator(noise_type=5)
    noise_generator_type_6 = NoiseGenerator(noise_type=6)
    noise_generator_type_7 = NoiseGenerator(noise_type=7)
    noise_generator_type_8 = NoiseGenerator(noise_type=8)

    # generate mask of noise
    img_mask_type_1 = noise_generator_type_1.generate_noise(
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.6, 0.7),
        noise_concentration=(0.7, 0.8),
    )
    img_mask_type_2 = noise_generator_type_2.generate_noise(
        noise_value=(0, 64),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.7, 0.8),
    )
    img_mask_type_3 = noise_generator_type_3.generate_noise(
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.5),
        noise_concentration=(0.5, 0.5),
    )
    img_mask_type_4 = noise_generator_type_4.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.9, 0.9),
        noise_concentration=(0.9, 0.9),
    )
    img_mask_type_5 = noise_generator_type_5.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_6 = noise_generator_type_6.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_7 = noise_generator_type_7.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_8 = noise_generator_type_8.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )

    elapsed_time = time() - start_time

    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    for i in range(8):
        plt.figure()
        eval("plt.imshow(img_mask_type_" + str(i + 1) + ", cmap='gray')")
        plt.title("noise mask type " + str(i + 1))

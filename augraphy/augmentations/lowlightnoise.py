"""
version: 0.0.1

Dependencies:
    - opencv :
    - numpy :

Documentation:
    - OpenCV : https://docs.opencv.org/4.x/
    - Numpy : https://numpy.org/doc/
    - Brightness Adjustment : https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html



"""
import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class LowLightNoise(Augmentation):
    """
    Simulate low-light conditions in an image by applying various noise effects.

    Args:
        num_photons_range (tuple): Range of the number of photons to simulate (default: (50, 100)).
        alpha_range (tuple): Range of alpha values for brightness adjustment (default: (0.2, 0.3)).
        beta_range (tuple): Range of beta values for brightness adjustment (default: (10, 30)).
        gamma_range (tuple): Range of gamma values for contrast adjustment (default: (1.0, 1.2)).
        bias_range (tuple): Range of bias values to add (default: (20, 40)).
        dark_current_value (float): Value for dark current simulation (default: 1.0).
        exposure_time (float): Length of the simulated exposure in seconds (default: 0.2).
        gain (float): Gain of the camera (default: 0.1).
        p (float): Probability of applying the augmentation (default: 1).

    """

    def __init__(
        self,
        num_photons_range=(50, 100),
        alpha_range=(0.7, 1.0),
        beta_range=(10, 30),
        gamma_range=(1, 1.8),
        bias_range=(20, 40),
        dark_current_value=1.0,
        exposure_time=0.2,
        gain=0.1,
        p=1,
    ):
        super().__init__(p=p)
        self.num_photons_range = num_photons_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.bias_range = bias_range
        self.dark_current_value = dark_current_value
        self.exposure_time = exposure_time
        self.gain = gain

    def _add_bias(self, image, value):
        """
        Add bias noise to the image.

        Args:
            image (numpy.ndarray): The input image.
            value (float): The bias value to add.

        """
        # setting seed for random number generation
        shape = image.shape
        columns = np.random.randint(0, shape[1], size=15)
        # adding constant offset to the image
        bias_im = np.zeros_like(image) + value
        # add random column noise to the image
        col_pattern = np.random.uniform(0, int(value * 0.1) // 2, size=shape[0])
        if len(shape) > 2:
            # skip alpha channel
            for channel in range(3):
                for column in columns:
                    bias_im[:, column, channel] = value + col_pattern
        else:
            for column in columns:
                bias_im[:, column] = value + col_pattern

        return bias_im

    def _apply_filter(self, image):
        """
        Apply a filter to the image.

        Args:
            image (numpy.ndarray): The input image.

        """
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(image, -1, kernel)
        return dst

    def _sensor_noise(self, image, current, exposure_time, value, gain=0.1):
        """
        Simulate sensor noise in the image.

        Args:
            image (numpy.ndarray): The input image.
            current (float): The dark current value.
            exposure_time (float): Length of the simulated exposure in seconds.
            value (float): The bias value.
            gain (float): The gain of the camera.
        """
        base_current = current * exposure_time / gain  # noise due to thermal heat of the sensor
        dark_im = np.random.poisson(base_current, size=image.shape)
        y_max, x_max = dark_im.shape[:2]
        n_hot = int(0.00001 * x_max * y_max)
        hot_x = np.random.randint(0, x_max, size=n_hot)
        hot_y = np.random.randint(0, y_max, size=n_hot)
        if len(dark_im.shape) > 2:
            for channel in range(3):
                dark_im[hot_y, hot_x, channel] = current * exposure_time / gain
        else:
            dark_im[hot_y, hot_x] = current * exposure_time / gain
        bias_im = self._add_bias(image.copy(), value)  # noise due to accumulation of photon on the screen
        noise = 0.1 * bias_im + 0.1 * dark_im
        return noise

    def _simulate_low_light_image(self, image, alpha, beta, gamma, bias, photons):
        """
        Simulate a low-light image with various noise effects.

        Args:
            image (numpy.ndarray): The input image.
            alpha (float): The alpha value for brightness adjustment.
            beta (float): The beta value for brightness adjustment.
            gamma (float): The gamma value for contrast adjustment.
            photons (int): The number of photons.
        """
        image = image.astype(np.float32)
        new_img = cv2.convertScaleAbs(
            image.copy(),
            alpha=alpha,
            beta=-beta,
        )  # reducing the brightness of the image by applying a linear function
        quantum_efficiency = random.random()
        noise = np.random.poisson(quantum_efficiency * photons, size=image.shape) + self._sensor_noise(
            image,
            self.dark_current_value,
            self.exposure_time,
            bias,
            self.gain,
        )
        noise_im = np.clip(new_img.copy() + noise, 0, 255).astype(np.uint8)
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):  # reducing the contrast using gamma adjustment
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(noise_im, lookUpTable)
        output_img = self._apply_filter(res)
        return output_img

    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()

            has_alpha = 0
            if len(result.shape) > 2 and result.shape[2] == 4:
                has_alpha = 1
                result, image_alpha = result[:, :, :3], result[:, :, 3]

            photons = random.randint(self.num_photons_range[0], self.num_photons_range[1])
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
            beta = random.uniform(self.beta_range[0], self.beta_range[1])
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            bias = random.uniform(self.bias_range[0], self.bias_range[1])
            result = self._simulate_low_light_image(
                result,
                alpha,
                beta,
                gamma,
                bias,
                photons,
            )

        if has_alpha:
            result = np.dstack((result, image_alpha))

        return result

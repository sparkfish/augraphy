import random

import cv2
import numpy as np

from augraphy.augmentations.lib import sobel
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class BadPhotoCopy(Augmentation):
    """Uses olsen noise to generate an effect of dirty copier
    :param layer: The image layer to apply the augmentation to.
    :type layer: string
    :param noise_density: Pair of floats determining density of noises.
            Lower value generates sparser noise.
    :type noise_density: tuple, optional
    :param max_iteration: Pair of ints determining the range of iterations for
            the requested noise value. Higher value generates sparser
            noise.
    :type max_iteration: tuple, optional
    :param hash_type: Types of hashes to generate olsen noise.
    :type hash_type: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        layer,
        noise_density=(0.1, 0.9),
        max_iteration=(7, 7),
        hash_type=0,
        p=0.5,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.layer = layer
        self.noise_density = noise_density
        self.max_iteration = max_iteration
        self.hash_type = hash_type
        self._SCALE_FACTOR = 2
        self.GAUSSIAN = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        self.BOX = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * 1 / 9
        self._blur_edge = 2  # extra pixels are needed for the blur (3 - 1).

        if self.hash_type == 1:
            self.noise_density = (0.01, 0.1)
        elif self.hash_type == 2:
            self.noise_density = (0.3, 0.3)
        elif self.hash_type == 3:
            self.noise_density = (0.01, 0.2)
        elif self.hash_type == 4:
            self.noise_density = (0.3, 0.5)
        elif self.hash_type == 5:
            self.noise_density = (0.1, 0.5)

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BadPhotoCopy({self.layer}, noise_density={self.noise_density}, max_iteration={self.max_iteration}, hash_type={self.hash_type}, p={self.p})"

    def noise(self, shape, position=None, iteration=7, kernel=None, transpose=True):
        """
        Returns a block of noise within the specific parameters.
        :param shape: shape of the noise block
        :param position: requested position within the noise.
        :param kwargs:  'iteration' number of iterations for the requested noise value.
                        'kernel'=GAUSSIAN, BOX use gaussian or box matrix.
                        'transpose'='True' transpose result.
        :return:
        """

        # check position (starting coordinates) of noise
        if position is None:
            position = [0] * len(shape)
        elif len(position) != len(shape):
            raise ValueError("Offset and shape values do not match")

        # initialize mask
        x, y = np.array(position)
        r_shape = self._required_dim(np.array(shape))
        pixels = np.zeros(r_shape, dtype="uint8")

        # check kernel
        if kernel is None:
            kernel = self.GAUSSIAN
        elif kernel.shape != (3, 3):
            raise NotImplementedError

        # apply olsen noise
        width, height = np.array(shape)
        self._olsen_noise(
            pixels,
            x,
            y,
            width,
            height,
            iteration=iteration,
            kernel=kernel,
        )

        # apply transpose
        if transpose:
            pixels = np.transpose(pixels[:width, :height])
        else:
            pixels = pixels[:width, :height]

        return pixels

    def _required_dim(self, dim):
        """
        Required Dim specifies the amount of extra edge pixels required to process the noise.
        The largest amount is the dim, plus both edge blur bytes, plus the extra scaling factor, and the shift of 1.
        :param dim:
        :return:
        """
        return dim + self._blur_edge + self._SCALE_FACTOR + 1

    def _olsen_noise(
        self,
        pixels,
        x,
        y,
        width,
        height,
        iteration=7,
        kernel=np.ones((3, 3)),
    ):
        """
        Olsen Noise generation algorithm.
        :param pixels: Pixel working space.
        :param x: x location to use for the chunk
        :param y: y location to use for the chunk
        :param width: width of the chunk
        :param height: height of the chunk
        :param iteration: iterations to apply to the noise.
        :return:
        """
        if iteration == 0:
            # Base case.
            self._apply_noise(pixels, x, y, width, height, iteration)
            return
        x_remainder = x & 1  # Adjust the x_remainder so we know how much more into the pixel are.
        y_remainder = y & 1

        self._olsen_noise(
            pixels,
            ((x + x_remainder) // self._SCALE_FACTOR) - x_remainder,
            ((y + y_remainder) // self._SCALE_FACTOR) - y_remainder,
            ((width + x_remainder) // self._SCALE_FACTOR) + self._blur_edge,
            ((height + y_remainder) // self._SCALE_FACTOR) + self._blur_edge,
            iteration - 1,
            kernel=kernel,
        )  # Recursive scope call.

        self._scale_shift(
            pixels,
            width + self._blur_edge,
            height + self._blur_edge,
            self._SCALE_FACTOR,
            x_remainder,
            y_remainder,
        )
        self._apply_kernel(pixels, width, height, kernel=kernel)
        self._apply_noise(pixels, x, y, width, height, iteration)

    def _scale_shift(self, pixels, width, height, factor, shift_x, shift_y):
        """
        Scale_shift pixels located in width and height of the array by the factor given and shifted by shift_x, and shift_y
        This process may be sped up applying np.kron or other accelerations later.
        :param pixels:
        :param width:
        :param height:
        :param factor:
        :param shift_x:
        :param shift_y:
        :return:
        """
        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                pixels[x, y] = pixels[(x + shift_x) // factor, (y + shift_y) // factor]

    def _apply_noise(
        self,
        pixels,
        x_within_field,
        y_within_field,
        width,
        height,
        iteration,
    ):
        for i, m in np.ndenumerate(pixels):
            hash_value = self._hash_random(
                i[0] + x_within_field,
                i[1] + y_within_field,
                iteration,
            )

            if self.hash_type != 4:
                hash_value &= 1 << (self.max_iteration[1] - iteration)

            pixels[i] += hash_value

    def _hash_random(self, *elements):
        """
        XOR hash the hashed values of each element, in elements
        :param elements: elements to be hashed and xor'ed together.
        :return:
        """
        hash_value = 0
        i = 0
        while i < len(elements):
            hash_value ^= elements[i]
            hash_value = self._hash(hash_value)
            i += 1
        return hash_value

    def _hash(self, v):
        value = int(v)
        original = value

        if self.hash_type == 1:  # noise start from page border
            value += original
            value ^= value << 1
            value = min(value, 255)

        elif self.hash_type == 2:  # disontinuous noises
            value += original * 5
            value ^= value << 2
            value = value % 255

        elif self.hash_type == 3:  # even noise on the whole page
            value = max(value, random.randint(190, 210))

        elif self.hash_type == 4:  # sparse and little noise
            value = 255 - random.randint(220, 230)

        elif self.hash_type == 5:  # uniform pattern
            value = 255 - (value * 4)
            value = value % 128

        else:  # totaly randomize
            q = value & 3

            if q == 3:
                value += original
                value ^= value << 32
                value ^= original << 36
                value += value >> 22
            elif q == 2:
                value += original
                value ^= value << 22
                value += value >> 34
            elif q == 1:
                value += original
                value ^= value << 20
                value += value >> 2
            value ^= value << 6
            value += value >> 10
            value ^= value << 8
            value += value >> 34
            value ^= value << 50
            value += value >> 12

        return value

    def _crimp(self, color):
        """
        crimps the values between 255 and 0. Required for some other convolutions like emboss where they go out of register.
        :param color: color to crimp.
        :return:
        """
        if color > 255:
            return 255
        if color < 0:
            return 0
        return int(color)

    def _apply_kernel(self, pixels, width, height, kernel=np.ones((3, 3))):
        """
        Applies a convolution with the results pixel in the upper left-hand corner.
        :param pixels:
        :param width:
        :param height:
        :param kernel:
        :return:
        """
        for index, m in np.ndenumerate(pixels[:width, :height]):
            pixels[index] = self._convolve(pixels, index, kernel)

    def _convolve(self, pixels, index, matrix):
        """
        Performs the convolution on that pixel by the given matrix. Note all values within the matrix are down and to the
        right from the current pixel. None are up or to the left. This is by design.
        :param pixels:
        :param index:
        :param matrix:
        :return:
        """
        parts = 0
        total = 0
        for mi, m in np.ndenumerate(matrix):
            parts += m  # keeps a running total for the parts.
            total += m * pixels[index[0] + mi[0], index[1] + mi[1]]
        if parts == 0:
            return self._crimp(total)
        return self._crimp(total // parts)

    def apply_augmentation(self, image):

        # get image dimensions
        if len(image.shape) > 2:
            ysize, xsize, dim = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            ysize, xsize = image.shape

        # initialize vectorized mapping functions
        # add random zero value noise
        add_noise_fn = lambda x: 0 if (x <= random.randint(0, 224)) else 255
        add_noise = np.vectorize(add_noise_fn)

        # remove noise by using min
        remove_noise_fn = (
            lambda original, noised, mask: min(original, noised + mask)
            if (np.uint(noised) + np.uint(mask) <= 255)
            else original
        )
        remove_noise = np.vectorize(remove_noise_fn)

        # add random noise range from 0-128
        add_edge_noise_fn = lambda x, y: random.randint(0, 128) if (y == 255 and random.random() < 0.70) else x
        add_edge_noise = np.vectorize(add_edge_noise_fn)

        image_sobel = sobel(image)
        image_sobel = cv2.GaussianBlur(image_sobel, (3, 3), 0)
        image_sobel[:, :][image_sobel[:, :] % 255 != 0] = 255
        image_sobel = cv2.dilate(image_sobel, (5, 5), iterations=2)

        image_sobel_sobel = sobel(image_sobel)
        image_sobel_sobel = cv2.dilate(image_sobel_sobel, (3, 3), iterations=2)
        image_sobel = add_edge_noise(image_sobel, image_sobel_sobel)
        image_sobel = cv2.GaussianBlur(image_sobel, (5, 5), 0)

        # create mask of noises
        shape = (
            random.randint(
                np.ceil(ysize * self.noise_density[0]),
                np.ceil(ysize * self.noise_density[1]),
            ),
            random.randint(
                np.ceil(xsize * self.noise_density[0]),
                np.ceil(xsize * self.noise_density[1]),
            ),
        )
        position = (random.randint(0, shape[1]), random.randint(0, shape[0]))
        iteration = random.randint(self.max_iteration[0], self.max_iteration[1])
        kernel = random.choice([self.GAUSSIAN, self.BOX])
        transpose = random.choice([True, False])

        # get mask of noise and resize it to image size
        mask = self.noise(shape, position, iteration, kernel, transpose)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # random flip mask vertically or horizontally
        if random.choice([True, False]):
            mask = cv2.flip(mask, 0)
        if random.choice([True, False]):
            mask = cv2.flip(mask, 1)

        # scale to 0-255
        if self.hash_type != 3 and self.hash_type != 4:
            mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255).astype("uint8")

        # randomize noise type
        noise_img_type = random.randint(0, 2)

        if self.hash_type == 3 or self.hash_type == 4:
            noise_img_type = 2

        noise_img = add_noise(mask).astype("uint8")

        if self.hash_type == 4:
            gaussian_kernel = (3, 3)
            for _ in range(random.randint(5, 10)):
                xloc = random.randint(0, xsize - 1)
                yloc = random.randint(0, ysize - 1)
                noise_img[yloc, xloc] = np.min(noise_img) / random.randint(1, 3)

        else:
            gaussian_kernel = (random.choice([3, 5, 7]), random.choice([3, 5, 7]))

        blurred = cv2.GaussianBlur(
            noise_img,
            gaussian_kernel,
            0,
        )

        # type 2 noise
        if noise_img_type == 2:
            noise_img = cv2.multiply(noise_img, blurred, scale=1 / 255)
            result = cv2.multiply(noise_img, image, scale=1 / 255)
            result = remove_noise(image, result, image_sobel)
            return result

        _, thresh1 = cv2.threshold(
            blurred,
            random.randint(64, 128),
            255,
            cv2.THRESH_BINARY,
        )

        # type 0 noise
        if noise_img_type == 0:
            result = cv2.multiply(thresh1, image, scale=1 / 255)
            result = remove_noise(image, result, image_sobel)
            return result

        grey_img = noise_img.copy()
        grey_img[:, :][grey_img[:, :] == 0] = random.choice(
            [255, 255, 255, random.randint(196, 224)],
        )

        # type 1 noise
        if noise_img_type == 1:
            noise_img = cv2.multiply(grey_img, thresh1, scale=1 / 255)
            result = cv2.multiply(noise_img, image, scale=1 / 255)
            result = remove_noise(image, result, image_sobel)
            return result

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data[self.layer][-1].result
            result = self.apply_augmentation(image)
            data[self.layer].append(AugmentationResult(self, result))

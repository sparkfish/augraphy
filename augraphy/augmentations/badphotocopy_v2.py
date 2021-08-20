import numpy as np
import random
import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class BadPhotoCopy_v2(Augmentation):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the range from which
           noise intensity is sampled.
    :type intensity: tuple, optional
    :param color_range: Pair of ints determining the range from which color
           noise is sampled.
    :type color_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self, p=0.5
    ):
        super().__init__(p=p)
        """Constructor method"""
        #super().__init__(p=p)

        self._SCALE_FACTOR = 2
        self.GAUSSIAN = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
        self._blur_edge = 2  # extra pixels are needed for the blur (3 - 1).

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"InkBleed(intensity_range={self.intensity_range}, color_range={self.color_range}, p={self.p})"

    def noise(self, shape, position=None, **kwargs):
        """
        Returns a block of noise within the specific parameters.
        :param shape: shape of the noise block
        :param position: requested position within the noise.
        :param kwargs:  'iteration'='0-7' number of iterations for the requested noise value.
                        'kernel'=GAUSSIAN, BOX use gaussian or box matrix.
                        'transpose'='True' transpose result.
        :return:
        """
        if position is None:
            position = [0] * len(shape)
        if len(position) != 2:
            raise NotImplementedError
        if len(shape) != 2:
            raise NotImplementedError
        if len(position) != len(shape):
            raise ValueError("Offset and shape values do not match")
        shape = np.array(shape)
        position = np.array(position)

        x, y = position
        r_shape = self._required_dim(shape)
        pixels = np.zeros(r_shape, dtype='uint8')
        iteration = 7
        try:
            iteration = kwargs['iteration']
        except KeyError:
            pass
        kernel = np.ones((3, 3))
        if 'kernel' in kwargs:
            kernel = kwargs['kernel']
        if kernel.shape != (3, 3):
            raise NotImplementedError
        width, height = shape
        self._olsen_noise(pixels, x, y, width, height, iteration=iteration, kernel=kernel)
        if 'transpose' in kwargs:
            return np.transpose(pixels[:width, :height])
        else:
            return pixels[:width, :height]


    def _required_dim(self, dim):
        """
        Required Dim specifies the amount of extra edge pixels required to process the noise.
        The largest amount is the dim, plus both edge blur bytes, plus the extra scaling factor, and the shift of 1.
        :param dim:
        :return:
        """
        return dim + self._blur_edge + self._SCALE_FACTOR + 1


    def _olsen_noise(self, pixels, x, y, width, height, iteration=7, kernel=np.ones((3, 3))):
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

        self._olsen_noise(pixels,
                    ((x + x_remainder) // self._SCALE_FACTOR) - x_remainder,
                    ((y + y_remainder) // self._SCALE_FACTOR) - y_remainder,
                    ((width + x_remainder) // self._SCALE_FACTOR) + self._blur_edge,
                    ((height + y_remainder) // self._SCALE_FACTOR) + self._blur_edge,
                    iteration - 1, kernel=kernel)  # Recursive scope call.

        self._scale_shift(pixels, width + self._blur_edge, height + self._blur_edge, self._SCALE_FACTOR, x_remainder, y_remainder)
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


    def _apply_noise(self, pixels, x_within_field, y_within_field, width, height, iteration):
        for i, m in np.ndenumerate(pixels[:width, :height]):
            pixels[i] += (self._hash_random(i[0] + x_within_field, i[1] + y_within_field, iteration) & (1 << (7 - iteration)))


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
    
    def sobel(self, image):
        """Computes the gradient of the image intensity function.

        :param image: The image over which to create an edge mask.
        :type image: numpy.array
        """
        gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient

    def apply_augmentation(self, image):
      add_noise_fn = (
          lambda x: 0
          if (x <= random.randint(0,224))
          else 255
      )
      add_noise = np.vectorize(add_noise_fn)
      
      remove_noise_fn = (
          lambda original, noised, mask: min(original, noised + mask)
          if (noised + mask <= 255)
          else original
      )

      add_edge_noise_fn = (
        lambda x, y: random.randint(0,128)
        if (y == 255 and random.random() < .70)
        else x
      )
      
      remove_noise = np.vectorize(remove_noise_fn)
      add_edge_noise = np.vectorize(add_edge_noise_fn)
      image_sobel = self.sobel(image)
      image_sobel = cv2.GaussianBlur(image_sobel, (3,3), 0)
      image_sobel[:,:][image_sobel[:,:]%255 != 0 ] = 255
      image_sobel = cv2.dilate(image_sobel, (5,5), iterations=5)

      image_sobel_sobel = self.sobel(image_sobel)
      image_sobel_sobel = cv2.dilate(image_sobel_sobel, (3,3), iterations=2)
      image_sobel = add_edge_noise(image_sobel, image_sobel_sobel)
      image_sobel = cv2.GaussianBlur(image_sobel, (5,5), 0)
      
      mask = self.noise((random.randint(64, 256),random.randint(64,256)), (random.randint(100,250), random.randint(100,250)), iteration=7, kernel=self.GAUSSIAN, transpose=True)
      mask = cv2.resize(mask, (image.shape[1],image.shape[0]))

      if (random.choice([True,False])):
        mask = cv2.flip(mask, 0)

      if (random.choice([True,False])):
        mask = cv2.flip(mask, 1)

      noise_img_type = random.randint(0, 1)
      noise_img = add_noise(mask).astype("uint8")

      # if (noise_img_type == 0):
      #     result = cv2.multiply(noise_img, image, scale=1/255)
      #     result = remove_noise(image, result, image_sobel)
      #     return result
      
      blurred = cv2.GaussianBlur(noise_img, (random.choice([3, 5, 7]),random.choice([3, 5, 7])), 0)
      
      # if (noise_img_type == 1):
      #     result = cv2.multiply(blurred, image, scale=1/255)
      #     result = remove_noise(image, result, image_sobel)
      #     return result
      
      # if (noise_img_type == 2):
      #     noise_img = cv2.multiply(noise_img, blurred, scale=1/255)
      #     result = cv2.multiply(noise_img, image, scale=1/255)
      #     result = remove_noise(image, result, image_sobel)
      #     return result
      
      _,thresh1 = cv2.threshold(blurred,random.randint(64,128),255,cv2.THRESH_BINARY)
      
      if (noise_img_type == 0):
          result = cv2.multiply(thresh1, image, scale=1/255)
          result = remove_noise(image, result, image_sobel)
          return result
      
      grey_img = noise_img.copy()
      grey_img[:,:][grey_img[:,:]==0] = random.choice([255, 255, 255, random.randint(196, 224)])
      
      if (noise_img_type == 1):
          noise_img = cv2.multiply(grey_img, thresh1, scale=1/255)
          result = cv2.multiply(noise_img, image, scale=1/255)
          result = remove_noise(image, result, image_sobel)
          return result
    
    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["ink"][-1].result
            result = self.apply_augmentation(image)
            data["ink"].append(AugmentationResult(self, result))


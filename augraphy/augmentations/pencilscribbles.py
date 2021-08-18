import numpy as np
import cv2
import random
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult

class PencilScribbles(Augmentation):
    """Applies random pencil scribbles to image.
    :param size_range: Pair of floats determining the range for
           the size of the scribble to be created
    :type size_range: tuple, optional
    :param count_range: Pair of floats determining the range for
           the number of scribbles to create.
    :type count_range: tuple, optional
    :param stroke_count_range: Pair of floats determining the range for
           the number of strokes to create in each scribble.
    :type stroke_count_range: tuple, optional
    :param thickness_range: Pair of floats determining the range for
           the size of the scribbles to create.
    :type thickness_range: tuple, optional
    :param brightness_change: Value change for the brightness of
           the strokes. Default 128 creates a graphite-like appearance.
           32 creates a charcoal-like appearance.
    :type brightness_change: int, optional
    :param p: Probability of this Augmentation being applied.
    :type p: float, optional
    """

    def __init__(
        self, size_range=(250, 400), count_range=(1, 10), stroke_count_range=(3, 6), thickness_range=(2, 6), brightness_change=128, p=0.5
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.size_range = size_range
        self.count_range = count_range
        self.stroke_count_range = stroke_count_range
        self.thickness_range = thickness_range
        self.brightness_change = brightness_change

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"PencilScribbles(size_range={self.size_range}, count_range={self.count_range}, stroke_count_range={self.stroke_count_range}, thickness_range={self.thickness_range}, brightness_change={self.brightness_change}, p={self.p})"

    def get_sobel(self, image):
        """Computes the gradient of the image intensity function.

        :param image: The image over which to create an edge mask.
        :type image: numpy.array
        """
        gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient

    def addNoise(self, image, intensity_range=(0.3, 0.5), color_range=(32, 128)):
        """Applies random noise to the input image.

        :param image: The image to noise.
        :type image: numpy.array
        :param intensity_range: Pair of bounds for intensity sample.
        :type intensity_range: tuple, optional
        :param color_range: Pair of bounds for 8-bit colors.
        :type color_range: tuple, optional
        """

        intensity = random.uniform(intensity_range[0], intensity_range[1])
        noise = (
            lambda x: random.randint(color_range[0], color_range[1])
            if (x == 0 and random.random() < intensity)
            else x
        )
        add_noise = np.vectorize(noise)

        return add_noise(image)

    def apply_pencil_stroke(self, stroke_image, image):
        apply_mask_fn = (
            lambda x, y: y if (x < 64) else x
        )
        apply_mask = np.vectorize(apply_mask_fn)
        stroke_image = cv2.cvtColor(stroke_image, cv2.COLOR_BGR2GRAY)
        noise_mask = self.addNoise(stroke_image, (0.3, 0.5), (32, 128))

        stroke_image = apply_mask(stroke_image, noise_mask)

        intensity = random.uniform(.4, .7)
        add_noise_fn = (
            lambda x, y: random.randint(32, 128)
            if (y == 255 and random.random() < intensity)
            else x
        )

        add_noise = np.vectorize(add_noise_fn)
        apply_mask = np.vectorize(apply_mask_fn)
        sobel = self.get_sobel(stroke_image)
        sobel = cv2.dilate(sobel, (5,5), iterations=3)
        stroke_image = add_noise(stroke_image, sobel)

        stroke_image = cv2.cvtColor(stroke_image, cv2.COLOR_GRAY2BGR)
        stroke_image = cv2.GaussianBlur(stroke_image, (3,3), 0)

        hsv = cv2.cvtColor(stroke_image.astype("uint8"), cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] += self.brightness_change
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        stroke_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return cv2.multiply(stroke_image, image, scale=1/255)

    def createScribble(self, max_height, max_width):
        size = random.randint(self.size_range[0], self.size_range[1])
        size = min([size, max_height, max_width])
        width, height = size, size  # picture's size

        strokes_img = np.zeros((height, width, 3), np.uint8) + 255 # make the background white

        for i in range(5):
            #lets say these are my black pixels in a white image.
            stroke_img = np.zeros((height, width, 3), np.uint8) + 255 # make the background white
            x = np.array([random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25)])
            y = np.array([random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25), random.randint(25, size-25)])

            start_stop = [random.randint(25, size//2), random.randint(size//2, size-25)]

            # Initilaize y axis
            lspace = np.linspace(min(start_stop), max(start_stop))

            #calculate the coefficients.
            z = np.polyfit(x, y, 2)

            #calculate x axis
            line_fitx = z[0]*lspace**2 + z[1]*lspace+ z[2]
            verts = np.array(list(zip(line_fitx.astype(int),lspace.astype(int))))
            cv2.polylines(stroke_img,[verts],False,(0,0,0),thickness=random.randint(self.thickness_range[0], self.thickness_range[1]))
            strokes_img = self.apply_pencil_stroke(stroke_img, strokes_img)

        return strokes_img

    def random_paste(self, paste, target):
        target_x = random.randint(0, target.shape[1]-paste.shape[1])
        target_y = random.randint(0, target.shape[0]-paste.shape[0])
        target[target_y:target_y+paste.shape[1],target_x:target_x+paste.shape[0]] = paste
        return target

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["post"][-1].result
            
            for i in range(random.randint(self.count_range[0], self.count_range[1])):
                scribbles = np.full(image.shape, 255).astype("uint8")
                strokes_img = self.createScribble(image.shape[1], image.shape[0])
                scribbles = self.random_paste(strokes_img, scribbles)
                image = cv2.multiply(scribbles, image, scale=1/255)

            data["post"].append(AugmentationResult(self, image))

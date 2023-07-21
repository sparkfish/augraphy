"""
version: 0.0.1
*********************************

Dependencies
- numpy
- PIL
- numba
- opencv

*********************************

References:

- Numpy Documentation: https://numpy.org/doc/
- PIL Documentation: https://pillow.readthedocs.io/en/stable/

- Numba Documentation: https://numba.readthedocs.io/en/stable/

- OpenCV Documentation:  https://docs.opencv.org/4.x/

- Voronoi Tessellation: a. https://en.wikipedia.org/wiki/Voronoi_diagram
                        b. https://www.generativehut.com/post/robots-and-generative-art-and-python-oh-my
                        c. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************

"""
import math
import os
import random
import warnings

import cv2
import numba as nb
import numpy as np
from numba import jit
from PIL import Image

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.meshgenerator import Noise
from augraphy.utilities.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class VoronoiTessellation(Augmentation):
    """
    This script generates a Voronoi Tessellation based on a set of random points in a plane. The tessellation
    is visualized by coloring or shading the region around each point with the color or shade of the corresponding
    random point. By default, Perlin Noise is added to the distances between each point and its closest random
    point to create a smoother, more organic looking tessellation.
    The class inherits methods and properties from the Augmentation base class.


    :param mult_range: range for amplification factor to generate Perlin noise , default lies between 50 and 80
    :type mult_range: tuple (int), optional
    :param seed: The seed value for generating the Perlin Noise, default value is 19829813472
    :type seed: int, optional
    :param num_cells_range: Range for the number of cells used to generate the Voronoi Tessellation. Default
                            lies between 1000 and 9000.
    :type num_cells_range: tuple (int), optional
    :param noise_type: If "random", integration of Perlin Noise in the pipeline is randomly selected.
                       If noise_type is "perlin", perlin noise is added to the background pattern,
                       otherwise no Perlin Noise is added.Perlin Noise is added to the image to create a smoother,
                       more organic looking tessellation.
    :type noise_type: string, optional
    :param background_value: Range for background color assigned to each point
    :type background_value: tuple (int)
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float


    """

    def __init__(
        self,
        mult_range=(50, 80),
        seed=19829813472,
        num_cells_range=(500, 1000),
        noise_type="random",
        background_value=(200, 256),
        p=1,
    ):
        super().__init__(p=p)
        self.mult_range = mult_range
        self.seed = seed
        self.num_cells_range = num_cells_range
        self.noise_type = noise_type
        self.background_value = background_value

    def __repr__(self):
        return f"Voronoi Tessellation: amplification_factor_range = {self.mult_range} , seed = {self.seed}, range of random points = {self.num_cells_range}, noise_type={self.noise_type}, background_value = {self.background_value}"

    @staticmethod
    @jit(nopython=True, cache=True)
    def generate_voronoi(width, height, num_cells, nsize, pixel_data, perlin_noise_2d):
        """
        Generates Voronoi Tessellation
        """
        img_array = np.zeros((width, height), dtype=np.uint8)
        for y in nb.prange(width):
            for x in nb.prange(height):
                dmin = math.hypot(height, width)
                for i in nb.prange(num_cells):
                    d = math.hypot(
                        (pixel_data[0][i] - x + perlin_noise_2d[0][x][y]),
                        (pixel_data[1][i] - y + perlin_noise_2d[1][x][y]),
                    )
                    if d < dmin:
                        dmin = d
                        j = i
                    nsize[j] += 1
                img_array[y][x] = pixel_data[2][j]
        return img_array

    def apply_augmentation(self):
        obj_noise = Noise()
        perlin_x = np.zeros((self.height, self.width))
        perlin_y = np.zeros((self.height, self.width))
        if self.perlin:
            perlin_x = np.array(
                [
                    [obj_noise.noise2D(x / 100, y / 100) * self.mult for y in range(self.height)]
                    for x in range(self.width)
                ],
            )
            perlin_y = np.array(
                [
                    [
                        obj_noise.noise2D((x + self.seed) / 100, (y + self.seed) / 100) * self.mult
                        for y in range(self.height)
                    ]
                    for x in range(self.width)
                ],
            )
        nx = [random.randrange(self.width) for _ in range(self.num_cells)]  # x-coordinates of random points
        ny = [random.randrange(self.height) for _ in range(self.num_cells)]  # y-coordinates of random points
        ng = [
            random.randrange(self.background_value[0], self.background_value[1]) for _ in range(self.num_cells)
        ]  # go through the number of cells and assign color
        nsize = np.zeros(self.num_cells, dtype=np.int32)
        img_array = self.generate_voronoi(
            self.width,
            self.height,
            self.num_cells,
            nsize,
            (nx, ny, ng),
            (perlin_x, perlin_y),
        )
        # try if it is able to save and read image properly, might facing permission issue
        try:
            image = Image.fromarray(img_array)
            image.save(os.getcwd() + "/Voronoi_example.png", "PNG", dpi=(300, 300))
            # reads it in a format so that it can be applied as a background pattern to the original image
            mesh = cv2.imread(os.getcwd() + "/Voronoi_example.png")
            os.remove(os.getcwd() + "/Voronoi_example.png")
        except Exception:
            mesh = img_array
        return mesh

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w = result.shape[:2]
            if self.noise_type == "random":
                self.perlin = random.choice([True, False])
            elif self.noise_type == "perlin":
                self.perlin = True
            else:
                self.perlin = False

            if self.perlin:
                self.width = self.height = random.choice(
                    [100, 120, 140, 160, 180, 200],
                )
                lst = [50, 70, 80, 90]
                find_random_divisor = (
                    lambda lst, b: random.choice([x for x in lst if x != 0 and b % x == 0])
                    if any(x != 0 and b % x == 0 for x in lst)
                    else 40
                )
                self.ws = find_random_divisor(
                    lst,
                    self.width,
                )
            else:
                self.width = self.height = random.choice(
                    [200, 210, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400],
                )
                lst = [100, 120, 140, 150, 160]
                find_random_divisor = (
                    lambda lst, b: random.choice([x for x in lst if x != 0 and b % x == 0])
                    if any(x != 0 and b % x == 0 for x in lst)
                    else 40
                )
                self.ws = find_random_divisor(
                    lst,
                    self.width,
                )

            self.mult = random.randint(self.mult_range[0], self.mult_range[1])
            self.num_cells = random.randint(self.num_cells_range[0], self.num_cells_range[1])
            voronoi_mesh = self.apply_augmentation()
            voronoi_mesh = cv2.resize(voronoi_mesh, (self.ws, self.ws), interpolation=(cv2.INTER_LINEAR))
            if len(image.shape) < 3 and len(voronoi_mesh.shape) > 2:
                voronoi_mesh = cv2.cvtColor(voronoi_mesh, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) > 2 and len(voronoi_mesh.shape) < 3:
                voronoi_mesh = cv2.cvtColor(voronoi_mesh, cv2.COLOR_GRAY2BGR)
            sw = PatternMaker()
            # to ensure the voronoi tessellation covers the whole image,
            # original image is padded and voronoi_mesh passes through it like a sliding window
            result = sw.make_patterns(result, voronoi_mesh, self.ws)
            result = result[self.ws : h + self.ws, self.ws : w + self.ws]
            return result

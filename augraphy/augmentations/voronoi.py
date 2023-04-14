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


    :param width: width of the tesellation, default is 200
    :type width: int
    :param height: height of the tessellation, default is 200
    :type height: int
    :param mult: range for amplification factor to generate Perlin noise , default lies between 50 and 80
    :type mult: int
    :param seed: The seed value for generating the Perlin Noise, default value is 19829813472
    :type seed: int
    :param num_cells: Range for the number of cells used to generate the Voronoi Tessellation. Default
                      lies between 1000 and 9000.
    :type num_cells: tuple (int)
    :param perlin: if True, Perlin Noise is added to the distance between each point and its closest random point to
    create a smoother, more organic looking tessellation. Default is True.
    :type perlin: bool
    :param background_value: Range for background color assigned to each point
    :type background_value: tuple (int)
    :param ws: patch height and width, default value is 200
    :type ws: int
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float


    """

    def __init__(
        self,
        width=200,
        height=200,
        mult=(50, 80),
        seed=19829813472,
        num_cells=(1000, 9000),
        perlin=True,
        background_value=(200, 256),
        ws=200,
        p=1,
    ):
        super().__init__(p=p)
        self.width = width
        self.height = height
        self.mult = random.randint(mult[0], mult[1])
        self.seed = seed
        self.num_cells = random.randint(num_cells[0], num_cells[1])
        self.perlin = perlin
        self.background_value = background_value
        self.ws = ws

    def __repr__(self):
        return f"Voronoi Tessellation: mesh_width = {self.width}, mesh_height={self.height}, amplification_factor = {self.mult} , seed = {self.seed}, number of random points = {self.num_cells}, perlin_noise={self.perlin}, background_value = {self.background_value}, window_size= {self.ws}"

    @staticmethod
    @jit(nopython=True, cache=True)
    def generate_voronoi(width, height, num_cells, nsize, pixel_data, perlin_noise_2d):
        """
        Generates Voronoi Tessellation Image
        """
        img_array = np.zeros((width, height), dtype=np.uint8)
        for y in range(width):
            for x in range(height):
                dmin = math.hypot(height, width)
                for i in range(num_cells):
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
        image = Image.fromarray(img_array)
        image.save("images/Voronoi_example.png", "PNG", dpi=(300, 300))
        mesh = cv2.imread("images/Voronoi_example.png")
        os.remove("images/Voronoi_example.png")
        return mesh

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w, _ = result.shape
            voronoi_mesh = self.apply_augmentation()
            voronoi_mesh = cv2.resize(voronoi_mesh, (self.ws, self.ws), interpolation=(cv2.INTER_LINEAR))
            sw = PatternMaker()
            # to ensure the voronoi tessellation covers the whole image,
            # original image is padded and voronoi_mesh passes through it like a sliding window
            result = sw.make_patterns(result, voronoi_mesh, self.ws)
            result = result[self.ws : h + self.ws, self.ws : w + self.ws, :]
            return result

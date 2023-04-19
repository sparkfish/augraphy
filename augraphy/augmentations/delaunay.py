"""
version: 0.0.1
*********************************

Dependencies
- numpy
- opencv

*********************************

References:

- Scipy Documentation: https://docs.scipy.org/doc/scipy/
- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation:  https://docs.opencv.org/4.x/

- Delaunay Tessellation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************

"""
import random
import warnings

import cv2
import numpy as np
from scipy import ndimage

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.meshgenerator import Noise
from augraphy.utilities.slidingwindow import PatternMaker

# from scipy.spatial import Delaunay

warnings.filterwarnings("ignore")


class DelaunayTessellation(Augmentation):
    """
    The Delaunay Tessellation is a method of dividing a geometric space
    into a set of triangles. This implementation generate a Delaunay Tessellation of an image with Perlin Noise by default to create smoother,
    more organic looking tessellations.
    The Delaunay Tessellation algorithm is a method of traingulating a set of points in a planar space such that the minimum angle of
    each triangle is maximized. This ensures that the triangles are as close to equilateral as possible.
    The Delaunay Tessellation is defined as the triangulation of a set of points such that no point is inside the circumcircle of any triangle.
    The algorithm works by iteratively adding points to the triangulation and re-triangulating the set of points after each point is added.
    The algorithm ensures that the triangulation remains a Delaunay tessellation by checking for the Delaunay condition after each point is added.
    The Delaunay Condition states that the circumcircle of each triangle in the triangulation must contain no other points in its interior.
    The class inherits methods and properties from the Augmentation base class.



    :param n_points_range: Range for the number of triangulating points from 500 to 800. Randomly selected.
    :type n_points_range: tuple (int), optional
    :param n_horizontal_points_range: Range for the number of points in the horizontal edge, from 500 to 800. The value is randomly selected.
    :type n_horizontal_points_range: tuple (int), optional
    :param n_vertical_points_range: Range for the number of points in the vertical edge, from 500 to 800. The value is randomly selected.
    :type n_vertical_points_range: tuple (int), optional
    :param noise_type: if "random", integration of Perlin Noise in the pipeline is randomly selected. Otherwise no Perlin Noise is added to the image.
    Perlin Noise is added to the image to create a smoother, more organic looking tessellation. .
    :type noise_type: string, optional
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float, optional

    """

    def __init__(
        self,
        n_points_range=(500, 800),
        n_horizontal_points_range=(500, 800),
        n_vertical_points_range=(500, 800),
        noise_type="random",
        p=1,
    ):
        super().__init__(p=p)
        self.n_points_range = n_points_range  # no. of random points generated on the geometric plane
        self.n_horizontal_points_range = n_horizontal_points_range  # no. of horizontal edge points
        self.n_vertical_points_range = n_vertical_points_range  # no. of edge vertical points
        self.noise_type = noise_type  # apply perlin or not

    def __repr__(self):
        return f"Delaunay Tessellation range of random points on geometric plane = {self.n_points_range}, range of horizontal edge points = {self.n_horizontal_points_range}, range of vertical edge points = {self.n_vertical_points_range}, noise_type = {self.noise_type}"

    def _edge_points(self, image):
        """
        Generate Random Points on the edge of an document image

        :param image: opencv image array
        :param length_scale: how far to space out the points in the goemetric
                             document image plane
        :param n_horizontal_points: number of points in the horizontal edge
                                    Leave as None to use length_scale to determine
                                    the value.
        :param n_vertical_points: number of points in the vertical edge
                                  Leave as None to use length_scale to determine
                                  the value
        :return: array of coordinates
        """
        ymax, xmax = image.shape[:2]
        if self.n_horizontal_points is None:
            self.n_horizontal_points = int(xmax / 200)

        if self.n_vertical_points is None:
            self.n_vertical_points = int(ymax / 200)

        delta_x = 4
        delta_y = 4

        return np.array(
            [[0, 0], [xmax - 1, 0], [0, ymax - 1], [xmax - 1, ymax - 1]]
            + [[delta_x * i, 0] for i in range(1, self.n_horizontal_points)]
            + [[delta_x * i, ymax] for i in range(1, self.n_horizontal_points)]
            + [[0, delta_y * i] for i in range(1, self.n_vertical_points)]
            + [[xmax, delta_y * i] for i in range(1, self.n_vertical_points)]
            + [[xmax - delta_x * i, ymax] for i in range(1, self.n_vertical_points + 30)],
        )

    def apply_augmentation(self):
        # Create an empty numpy array of zeros with the given size
        img = np.ones((self.height, self.width, 3), np.uint8) * 255
        # Define some points to use for the Delaunay triangulation
        points = np.array(
            [(random.uniform(0, self.width), random.uniform(0, self.height)) for i in range(self.n_points)],
        )
        points = np.concatenate([points, self._edge_points(img)])
        # Perform the Delaunay triangulation on the points
        rect = (0, 0, self.width, self.height)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            if p[0] >= 0 and p[0] < self.width and p[1] >= 0 and p[1] < self.height:
                subdiv.insert((int(p[0]), int(p[1])))

        triangles = subdiv.getTriangleList()
        triangles = triangles.astype(np.int32)
        # adding perlin noise
        if self.perlin:
            obj_noise = Noise()
            noise = np.array(
                [
                    [obj_noise.noise2D(j / 200, i / 200) * 50 + 200 for j in range(self.width)]
                    for i in range(self.height)
                ],
                np.float32,
            )
            # noise = np.array((noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 255 , np.uint8)
            nh, nw = noise.shape
            # Convert the blue texture to grayscale
            gray_texture = np.dot(noise[..., :3], [0.299, 0.587, 0.114])
            white_texture = np.zeros((nh, nw, 3), dtype=np.uint8)
            white_texture[..., 0] = gray_texture
            white_texture[..., 1] = gray_texture
            white_texture[..., 2] = gray_texture
            img = cv2.addWeighted(
                white_texture,
                0.1,
                img,
                0.9,
                0,
            )  # creating a white texture from the perlin noise mesh
            img = ndimage.gaussian_filter(img, sigma=(3, 3, 0), order=0)  # applying gaussian filter
            colors = [
                (250, 235, 215),
                (240, 240, 230),
                (253, 245, 230),
                (255, 245, 238),
                (255, 248, 220),
                (248, 248, 255),
                (255, 240, 245),
                (245, 255, 250),
                (255, 250, 250),
                (240, 248, 255),
                (240, 255, 255),
                (240, 255, 240),
                (255, 245, 238),
                (243, 229, 171),
                (250, 250, 210),
            ]
            alt_colors = [
                (255, 255, 240),
                (255, 250, 205),
                (238, 232, 170),
                (255, 255, 224),
                (255, 239, 213),
            ]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Draw the Delaunay triangulation on the empty numpy array

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                if (
                    pt1[0]
                    and pt2[0]
                    and pt3[0] <= self.width * 0.80
                    and (pt1[0] and pt2[0] and pt3[0] >= self.width * 0.40)
                ):
                    color = colors[np.random.randint(len(colors))]  # choose from colors

                elif pt1[0] and pt2[0] and pt3[0] <= self.width * 0.40:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                else:
                    color = alt_colors[np.random.randint(len(alt_colors))]

                cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3]), color)
        else:
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                colors = [
                    (250, 235, 215),
                    (240, 240, 230),
                    (253, 245, 230),
                    (255, 245, 238),
                    (255, 248, 220),
                    (248, 248, 255),
                    (255, 240, 245),
                    (245, 255, 250),
                    (255, 250, 250),
                    (240, 248, 255),
                    (240, 255, 255),
                    (240, 255, 240),
                    (255, 245, 238),
                    (243, 229, 171),
                    (250, 250, 210),
                ]
                alt_colors = [
                    (255, 255, 240),
                    (255, 250, 205),
                    (238, 232, 170),
                    (255, 255, 224),
                    (255, 239, 213),
                ]
                if (
                    pt1[0]
                    and pt2[0]
                    and pt3[0] <= self.width * 0.80
                    and (pt1[0] and pt2[0] and pt3[0] >= self.width * 0.40)
                ):
                    color = colors[np.random.randint(len(colors))]  # choose from colors

                elif pt1[0] and pt2[0] and pt3[0] <= self.width * 0.40:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                else:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                color = colors[np.random.randint(len(colors))]  # choose from colors
                cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3]), color)
        return img

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            self.width = self.height = random.choice(
                [400, 480, 500, 600, 640, 720],
            )  # randomly selecting the width and the height of the background pattern
            self.n_points = random.randint(
                self.n_points_range[0],
                self.n_points_range[1],
            )  # randomly selecting the number of points in the geometric plane
            self.n_horizontal_points = random.randint(
                self.n_horizontal_points_range[0],
                self.n_horizontal_points_range[1],
            )  # randomly selecting the edge horizontal points in the goemetric plane
            self.n_vertical_points = random.randint(
                self.n_vertical_points_range[0],
                self.n_vertical_points_range[1],
            )  # randonly selecting the edge vertical points in the geometric plane
            if self.noise_type == "random":
                self.perlin = random.choice(
                    [True, False],
                )  # randomly select to apply Perlin Noise on top of the Tessellation
            else:
                self.perlin = False
            lst = [100, 120, 160]
            find_random_divisor = (
                lambda lst, b: random.choice([x for x in lst if x != 0 and b % x == 0])
                if any(x != 0 and b % x == 0 for x in lst)
                else None
            )
            self.ws = find_random_divisor(
                lst,
                self.width,
            )  # finding the window size for the patch, which will be passed over the original image like a Sliding-Window
            result = image.copy()
            h, w = result.shape[:2]
            delaunay_mesh = self.apply_augmentation()
            threshold = self.ws // 20
            delaunay_mesh = delaunay_mesh[threshold : h - threshold, threshold : w - threshold]
            delaunay_mesh = cv2.resize(delaunay_mesh, (self.ws, self.ws), interpolation=cv2.INTER_LINEAR)
            sw = PatternMaker(alpha=0.49)
            result = sw.make_patterns(image=result, mesh_img=delaunay_mesh, window_size=self.ws)
            result = result[self.ws : h + self.ws, self.ws : w + self.ws]
            return result

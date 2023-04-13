'''
version: 0.0.1
*********************************

Dependencies
- scipy
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

'''
import random 
import cv2
import numpy as np
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.meshgenerator import Noise
from augraphy.utilities.slidingwindow import PatternMaker
import warnings
warnings.filterwarnings("ignore")


class DelaunayTessellation(Augmentation):
    '''
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

    :param n_points: Range for the number of triangulating points
    :type n_points: tuple (int)
    :param n_horizontal_points: Range for the number of points in the horizontal edge, default = 200
    :type n_horizontal_points: tuple (int)
    :param n_vertical_points: Range for the number of points in the vertical edge, default = 200
    :type n_vertical_points: tuple (int)
    :param perlin: if True, Perlin Noise is added to the distance between each point and its closest random point to 
    create a smoother, more organic looking tessellation. Default is True.
    :type perlin: bool
    :param ws: patch height and width, default value is 200
    :type ws: int
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float 
    
    '''
    def __init__(self, width = 500, height = 500, n_points = (500, 800), n_horizontal_points=(5, 50), n_vertical_points= (5, 50),  perlin = True, ws = 200, p =1):
        super().__init__(p=p)
        self.width = width
        self.height = height
        self.n_points = random.randint(n_points[0], n_points[1])
        self.n_horizontal_points = random.randint(n_horizontal_points[0], n_horizontal_points[1])
        self.n_vertical_points = random.randint(n_vertical_points[0], n_vertical_points[1])
        self.perlin = perlin
        self.ws = ws


    def __repr__(self):
        return f"Delaunay Tessellation pattern_width = {self.width}, pattern_height= {self.height}, no. of random points on geometric plane = {self.n_points}, no. of horizontal edge points = {self.n_horizontal_points}, no. of vertical edge points = {self.n_vertical_points}, perlin_noise = {self.perlin}, window_size = {self.ws}"
        
    def _edge_points(self, image):
        '''
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
        '''
        ymax, xmax = image.shape[:2]
        if self.n_horizontal_points is None:
            self.n_horizontal_points = int(xmax / 200)

        if self.n_vertical_points is None:
            self.n_vertical_points = int(ymax / 200)

        delta_x = xmax / self.n_horizontal_points
        delta_y = ymax / self.n_vertical_points

        return np.array(
              [[0, 0], [xmax, 0], [0, ymax], [xmax, ymax]]
              + [[delta_x * i, 0] for i in range(1, self.n_horizontal_points)]
              + [[delta_x * i, ymax] for i in range(1, self.n_horizontal_points)]
              + [[0, delta_y * i] for i in range(1, self.n_vertical_points)]
              + [[xmax, delta_y * i] for i in range(1, self.n_vertical_points)]
            )
    
    
    def apply_augmentation(self):
        # Create an empty numpy array of zeros with the given size
        img = np.ones((self.height, self.width, 3), np.uint8) 
        # Define some points to use for the Delaunay triangulation
        points = np.array([(random.uniform(0, self.width), random.uniform(0, self.height)) for i in range(self.n_points)]) 
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
            img = [[obj_noise.noise2D(j/200 , i/200)*50 + 100  for j in range(self.width)] for i in range(self.height)]
            img = np.array(img, np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Draw the Delaunay triangulation on the empty numpy array
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                cv2.line(img, pt1, pt2, (255, 255, 255), 1)
                cv2.line(img, pt2, pt3, (255, 255, 255), 1)
                cv2.line(img, pt3, pt1, (255, 255, 255), 1)
        else:
            img = img*255
            min_val = 150
            max_val = 255
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                color = (np.random.randint(min_val, max_val), np.random.randint(min_val, max_val), np.random.randint(min_val, max_val))
                cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3]), color)
        return img







    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w, _ = result.shape
            delaunay_mesh = self.apply_augmentation()
            if self.perlin:
                crop_thresh = delaunay_mesh.shape[0]//50
            else:
                crop_thresh = delaunay_mesh.shape[0]//20
            delaunay_mesh = delaunay_mesh[crop_thresh: h -crop_thresh, crop_thresh: w-crop_thresh, :]
            delaunay_mesh = cv2.resize(delaunay_mesh, (self.ws,self.ws), interpolation=cv2.INTER_LINEAR)
            sw = PatternMaker()
            result = sw.make_patterns(result, delaunay_mesh, self.ws)
            result = result[self.ws: h+ self.ws, self.ws: w +self.ws, :]
            return result


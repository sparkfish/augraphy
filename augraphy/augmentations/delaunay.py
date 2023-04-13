'''
version: 0.0.1
*********************************

Dependencies
- scipy
- matplotlib
- numpy
- opencv

*********************************

References:

- Scipy Documentation: https://docs.scipy.org/doc/scipy/

- Matplotlib Documentation: https://matplotlib.org/stable/index.html

- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation:  https://docs.opencv.org/4.x/

- Delaunay Tessellation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************

'''

import random 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from augraphy.base.augmentation import Augmentation
from augraphy.utilities.meshgenerator import Noise
from augraphy.utilities.slidingwindow import PatternMaker
import os
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

    Email: sxs200326@utdallas.edu
    
    
    '''
    def __init__(self, n_points = (500, 800), n_horizontal_points=(5, 50), n_vertical_points= (5, 50),  perlin = True, ws = 200, p =1):
        super().__init__(p=p)
        self.n_points = random.randint(n_points[0], n_points[1])
        self.n_horizontal_points = random.randint(n_horizontal_points[0], n_horizontal_points[1])
        self.n_vertical_points = random.randint(n_vertical_points[0], n_vertical_points[1])
        self.perlin = perlin
        self.ws = ws


    def __repr__(self):
        return f"Delaunay Tessellation no. of random points on geometric plane = {self.n_points}, no. of horizontal edge points = {self.n_horizontal_points}, no. of vertical edge points = {self.n_vertical_points}, perlin_noise = {self.perlin}, window_size = {self.ws}"
        
    def _edge_points(self, image):
        '''
        Generate Random Points on the edge of an document image

        :param image: opencv image array
        :
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
        # Generate a set of random points
        im = np.zeros((1, 1), dtype = np.uint) * 255
        points = np.array([(random.uniform(0, 1), random.uniform(0, 1)) for i in range(self.n_points)]) 
        points = np.concatenate([points, self._edge_points(im)])
        if self.perlin:
            obj_noise = Noise()
            # Compute the Perlin noise values for each point
            noise_vals = np.array([obj_noise.noise2D(x*10, y*10) for x, y in points])

        # Compute the Delaunay triangulation of the points
        tri = Delaunay(points)

        # Create a colormap based on grayscale
        cmap_name = 'my_cmap'
        cmap = LinearSegmentedColormap.from_list(cmap_name, [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0)], N=256)
        mpl.rcParams['savefig.pad_inches'] = 0
        # Plot the triangles
        _, ax = plt.subplots(figsize=(4,4))
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0, y=0)
        for _, simplex in enumerate(tri.simplices):
            if self.perlin:
                avg_noise = np.mean(noise_vals[simplex])
                color = cmap(avg_noise)
            else:
                color  = cmap(0)
            ax.add_patch(plt.Polygon(points[simplex], fill=True, edgecolor='white', facecolor=color))
        ax.plot(points[:,0], points[:,1], '.', color='white')
        plt.savefig('images/delaunay_perlin.png', dpi=300, bbox_inches='tight')
        mesh = cv2.imread("images/delaunay_perlin.png")
        # cv2.imshow("Mesh image", mesh)
        # cv2.waitKey(0)
        os.remove("images/delaunay_perlin.png")
        return mesh


    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            result = image.copy()
            h, w, _ = result.shape
            delaunay_mesh = self.apply_augmentation()
            delaunay_mesh = delaunay_mesh[10: h -10, 10: w-10, :]
            delaunay_mesh = cv2.resize(delaunay_mesh, (self.ws,self.ws), interpolation=cv2.INTER_LINEAR)
            sw = PatternMaker()
            result = sw.make_patterns(result, delaunay_mesh, self.ws)
            result = result[self.ws: h+ self.ws, self.ws: w +self.ws, :]
            return result


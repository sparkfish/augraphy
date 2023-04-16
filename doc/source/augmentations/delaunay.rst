**********
DelaunayTessellation
**********

.. autoclass:: augraphy.augmentations.voronoi.DelaunayTessellation
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The Delaunay augmentation applies Delaunay Tessellation with Perlin Noise by default, creating organic looking background patterns.
Initially, a clean image with a single line of text is created.

Code example:

::

    # import libraries
    import cv2
    import numpy as np
    from augraphy import *


    # create a clean image with single line of text
    image = np.full((500, 1500,3), 250, dtype="uint8")
    cv2.putText(
        image,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        (80, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        3,
    )

    cv2.imshow("Input image", image)

Clean image:

.. figure:: augmentations/input.png

---------
Example 1
---------
In this example, a VoronoiTessellation instance with  Perlin Noise set to True is initialized, a background pattern is created and the patch is passed through the original image like a Sliding Window.
The number of points on the geometric plane is set in between 500 and 800. The number of horizontal edge points is set between 1 and 50.
The number of vertical points is set between 1 and 50. The shape (h,w) of the pattern patch is set to 20 which passes through the Original Image like a Sliding Window.


Code example:

::

    delaunay_pattern = DelaunayTessellation(
                                        width = 500,
                                        height = 500,
                                        n_points = (400, 800),
                                        n_horizontal_points=(1, 50),
                                        n_vertical_points=(1, 50),
                                        perlin = True ,
                                        ws = 200,
                                        color_background=(200,255))

    img_final = delaunay_pattern(image)
    cv2.imshow("Delaunay Image", img_final)


Augmented image:

.. figure:: augmentations/delaunay/delaunay.png

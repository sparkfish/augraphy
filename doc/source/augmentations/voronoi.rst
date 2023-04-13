**********
VoronoiTessellation
**********

.. autoclass:: augraphy.augmentations.voronoi.VoronoiTessellation
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The Voronoi augmentation applies Voronoi Tessellation with Perlin Noise by default, creating organic looking background patterns.
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
Pattern width and height is 200 and the pattern background value for each randomly selected point is set to bright value (250, 256).
The number of random points generated is set in between 1000 and 9000.
Amplification factor for Perlin Noise is set in between 50 and 80 and the seed is set to 19829813472. 
The shape (h,w) of the pattern patch is set to 40 which passes through the Original Image like a Sliding Window.


Code example:

::

    voronoi_pattern = VoronoiTessellation(width=200,
                             height = 200,
                             mult = (50,80),
                             seed = 19829813472 ,
                             num_cells = (1000,9000),
                             perlin = True,
                             background_value = (200, 256), 
                             ws = 40
                            )
                             
    img_voronoi = voronoi_pattern(image)
    cv2.imshow("Voronoi Pattern", img_voronoi)


Augmented image:

.. figure:: augmentations/voronoi/voronoi.png

*********
DotMatrix
*********

.. autoclass:: augraphy.augmentations.dotmatrix.DotMatrix
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The DotMatrix augmentation creates dot matrix effect by drawing dots of mean color in the detected contours.

Initially, a clean image with single line of text is created.

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
In this example, a DotMatrix augmentation instance is initialized and the shape of dot matrix effect is set to circle ("circle").
The width and height of each dot in dot matrix effect is set to 5 pixels (5,5).
The min and max width of the contours to apply the effect are 1 (1,1) and 50 (50,50) respectiely.
The min and max height of the contours to apply the effect are 1 (1,1) and 50 (50,50) respectiely.
The min and max area of the contours to apply the effect are 10 (10,10) and 800 (800,800) respectiely.
The median kernel value in removing the existing contours is set to 29 (29,29).
No Gaussian blurring is applied to the dot matrix effect (1,1).
There is no rotation in each dot of the dot matrix effect too (0,0).

Code example:

::

    dotmatrix = DotMatrix(dot_matrix_shape="circle",
          	      	  dot_matrix_dot_width_range=(5, 5),
                	  dot_matrix_dot_height_range=(5, 5),
                	  dot_matrix_min_width_range=(1, 1),
                	  dot_matrix_max_width_range=(50, 50),
                	  dot_matrix_min_height_range=(1, 1),
                	  dot_matrix_max_height_range=(50, 50),
                	  dot_matrix_min_area_range=(10, 10),
                	  dot_matrix_max_area_range=(800, 800),
                	  dot_matrix_median_kernel_value_range = (29,29),
                	  dot_matrix_gaussian_kernel_value_range=(1, 1),
                	  dot_matrix_rotate_value_range=(0, 0)
			  )


    img_dotmatrix = dotmatrix(image)
    cv2.imshow("dotmatrix", img_dotmatrix)

Augmented image:

.. figure:: augmentations/dotmatrix/dot_matrix.png

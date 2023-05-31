**********
PageBorder
**********

.. autoclass:: augraphy.augmentations.pageborder.PageBorder
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The Page Border augmentation stacking multiple images, creating an effect of single or multiple page borders on any side of the page.

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
In this example, a PageBorder augmentation instance is initialized and the page borders effect width and height is set to 30 and 40 pixels (30,40).
The page border color is set to black (0,0,0) and the background color is set to white (255,255,255).
The number of pages is set to 5 (5) and there is no rotation in the page border effect (0,0).
The curvy frequency in the borders is set to random value between 2 and 3 (2,3), each with height of 2 to 4 pixels (2,4).
The curvy one side length is set to random value in between 50 to 100 (50, 100).
The page border effect is applied within the page (same_page_border=1) so that no padding will be done.

Code example:

::

    page_border = PageBorder(page_border_width_height = (30, 40),
                             page_border_color=(0, 0, 0),
			     page_border_background_color=(255, 255, 255),
                             page_numbers = 5,
                             page_rotation_angle_range = (0, 0),
                             curve_frequency=(2, 3),
                             curve_height=(2, 4),
                             curve_length_one_side=(50, 100),
                             same_page_border=1,
                             )
    img_page_border = page_border(image)
    cv2.imshow("page_border", img_page_border)


Augmented image:

.. figure:: augmentations/pageborder/page_border.png

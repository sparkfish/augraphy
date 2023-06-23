***********
BookBinding
***********

.. autoclass:: augraphy.augmentations.bookbinding.BookBinding
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The BookBinding augmentation creates the effect of a page curling away from the scanner bed, towards the binding of a book or magazine. Right now it supports 2 variants of page curling direction.


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
In this example, a BookBinding augmentation instance is initialized and the curling direction is set to bend up (0).
The shadow radius range is set to 50 to 100% of the page (50,100).
The amount of page curling in pixels is set to between 200 and 300 (200,300).
The percentage of page mirroring for the binding effect is set to 20-30% (0.2, 0.3) of the input image.
The backdrop color of the effect is set to white color (255, 255, 255).

Code example:

::

    book_binder_up = BookBinding(radius_range=(50, 100),
                                 curve_range=(200, 300),
                                 mirror_range=(0.2, 0.3),
                                 curling_direction=0,
                                 backdrop_color=(255,255,255),
                                 )

    img_book_binding_up= book_binder_up(image)
    cv2.imshow("book_binding_up", img_book_binding_up)

Augmented image:

.. figure:: augmentations/bookbinding/book_binding_up.png

---------
Example 2
---------
In this example, a BookBinding augmentation instance is initialized and the curling direction is set to bend down (1).
The shadow radius range is set to 50 to 100% of the page (50,100).
The amount of page curling in pixels is set to between 200 and 300 (200,300).
The percentage of page mirroring for the binding effect is set to 20-30% (0.2, 0.3) of the input image.
The backdrop color of the effect is set to white color (255, 255, 255).

Code example:

::

    book_binder_down = BookBinding(radius_range=(50, 100),
                                   curve_range=(200, 300),
                                   mirror_range=(0.2, 0.3),
                                   curling_direction=1,
                                   backdrop_color=(255,255,255),
                                   )

    img_book_binding_down= book_binder_down(image)

Augmented image:

.. figure:: augmentations/bookbinding/book_binding_down.png

***************
PencilScribbles
***************

.. autoclass:: augraphy.augmentations.pencilscribbles.PencilScribbles
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The Pencil Scribbles augmentation applies random pencil scribbles to image.

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
In this example, a PencilScribbles augmentation instance is initialized and the scribbles type is set to "random", where it selects "lines" or "text" based scribbles randomly.
The location of scribbles is "random" and the size of scribbles effect is set in between 250 to 600 pixels (250, 600).
Number of scribbles is set in between 2 to 3 (2,3) and the thickness of scribbles is set to random value in between 1 and 3 (1,3).
The brightness change of scribbles effect is set to select randomly from 32, 64 and 128 [32. 64. 128].
If the scribbles type is "text", the scribbles text value and font type is set to "random" where it selects randomly from the internal settings.
Additionally, the scribbles text rotate range is set to rotate randomly in any angle between 0 to 360 degree (0, 360).
If the scribbles type is "lines", the stroke count for each scribbles is set at random value in between 1 and 6 (1, 6).

Code example:

::

    pencil_scribbles = PencilScribbles(scribbles_type="random",
                                       scribbles_location="random",
                                       scribbles_size_range=(250, 600),
                                       scribbles_count_range=(2, 3),
                                       scribbles_thickness_range=(1, 3),
                                       scribbles_brightness_change=[32, 64, 128],
                                       scribbles_text="random",
                                       scribbles_text_font="random",
                                       scribbles_text_rotate_range=(0, 360),
                                       scribbles_lines_stroke_count_range=(1, 6)
				       )

    img_pencil_scribbles = pencil_scribbles(image)
    cv2.imshow("pencil_scribbles", img_pencil_scribbles)


Augmented image:

.. figure:: augmentations/pencilscribbles/pencil_scribbles.png

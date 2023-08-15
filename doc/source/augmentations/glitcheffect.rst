************
GlitchEffect
************

.. autoclass:: augraphy.augmentations.glitcheffect.GlitchEffect
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The GlitchEffect augmentation create glitch effect by applying ColorShift and shifts patches of image horizontally or vertically.

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
In this example, a GlitchEffect augmentation instance is initialized and glitch direction is set to horizontal ("horizontal").
The number of glitch effect is set to any random number between 8 and 16 (8,16).
The glitch size is set to random value in between 5 and 50 pixels (5, 50).
The glitch offset is set of random value in between 5 and 10 pixels (5, 10).

Code example:

::

    glitcheffect= GlitchEffect(glitch_direction = "horizontal",
                               glitch_number_range = (8, 16),
                               glitch_size_range = (5, 50),
                               glitch_offset_range = (5, 10)
                               )

    img_glitcheffect = glitcheffect(image)

    cv2.imshow("glitcheffect, img_glitcheffect)

Augmented image:

.. figure:: augmentations/glitcheffect/glitch_effect.png

************
BleedThrough
************

.. autoclass:: augraphy.augmentations.bleedthrough.BleedThrough
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The BleedThrough augmentation creates ink bleed through page effect from random image in cache folder or the reverse side of image if there's no image in the cache directory.

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
In this example, a BleedThrough augmentation instance is initialized.
The bleeding intensity is set to moderate intensity range (0.1, 0.2).
The bleed kernel size is set to large value (17, 17) to enable a larger bleeding effect.
The alpha value is set to low value (0.3) so that the bleeding effect won't be too distinct.
The offsets value is set to random values between 10 and 20 (10,20) so that the bleeding ink effect will not stacked with the text in input image.

Code example:

::

    bleedthrough = BleedThrough(intensity_range=(0.1, 0.2),
                                color_range=(0, 224),
                                ksize=(17, 17),
                                sigmaX=0,
                                alpha=0.3,
                                offsets=(10, 20),
                            )

    img_bleedthrough = bleedthrough(image)
    cv2.imshow("bleedthrough", img_bleedthrough)

Augmented image:

.. figure:: augmentations/bleedthrough/bleedthrough.png

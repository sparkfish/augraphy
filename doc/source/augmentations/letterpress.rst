***********
Letterpress
***********

.. autoclass:: augraphy.augmentations.letterpress.Letterpress
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The Letterpress augmentation produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

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
In this example, a Letterpress augmentation instance is initialized and the number of noise cluster is set randomly in between 300 and 800 (300, 800).
In each of the cluster, number of samples is set randomly in between 200 and 500 (200, 500) and the spreadness of samples is set to large value (1500, 5000).
The noise value is set to random value in between 200 and 255 (200, 255) and blurring effect is enabled (1).

Code example:

::

    letterpress = Letterpress(n_samples=(200, 500),
                              n_clusters=(300, 800),
                              std_range=(1500, 5000),
                              value_range=(200, 255),
                              value_threshold_range=(128, 128),
                              blur=1
                              )
    
    img_letterpress = letterpress(image)
    cv2.imshow("letterpress", img_letterpress)


Augmented image:

.. figure:: augmentations/letterpress/letterpress.png

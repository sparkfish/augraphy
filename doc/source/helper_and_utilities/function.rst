********
Function
********

.. autoclass:: augraphy.utilities.function.Function
    :members:
    :undoc-members:
    :show-inheritance:


--------
Overview
--------
Function allows a custom defined function to be added into the augmentation pipeline.

-------
Example
-------
In this example, Function is use to add a manually defined function into the augmentation pipeline.
::

    # import libraries
    from augraphy import *
    import cv2
    import numpy as np
    from augraphy.utilities.function import Function

    # custom function
    def invert_colour(image):
        return np.invert(image.astype("uint8"))

    # initialize phases and pipeline
    ink_phase   = [Function(fs=invert_colour,p=1)]
    paper_phase = []
    post_phase  = []

    # initialize pipeline
    pipeline    = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    # create input image
    image = np.full((1200, 1200,3), 250, dtype="uint8")
    cv2.putText(
        image,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        (80, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        0,
        3,
    )

    # augment image
    augmented_image = pipeline.augment(image)["output"]



Input image:

.. figure:: input/input.png

Augmented image:

.. figure:: function/output.png

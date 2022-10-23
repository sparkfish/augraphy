*****
OneOf
*****

.. autoclass:: augraphy.base.oneof.OneOf
    :members:
    :undoc-members:
    :show-inheritance:


--------
Overview
--------
OneOf function allows augmentation pipeline to select an augmentation randomly from the list.

-------
Example
-------
In this example, OneOf function is use to select one augmentation randomly in ink phase of the augmentation pipeline.
In post phase, two OneOf functions are nested inside another OneOf function to select augmentation randomly.::

    # import libraries
    from augraphy import *
    import cv2
    import numpy as np

    # initialize phases and pipeline
    # ink phase with single usage of OneOf function
    ink_phase   = [
            OneOf([

                    InkBleed(p=1),
                    WaterMark(p=1)

                   ],p=1)
            ]

    paper_phase = []

    # post phase with two nested OneOf function inside another OneOf function.
    post_phase  = [
            OneOf([

                    OneOf([

                            BleedThrough(p=1),
                            DirtyDrum(p=1)

                            ],p=1),
                    OneOf([

                            Folding(p=1),
                            DirtyRollers(p=1)

                            ],p=1)

                    ],p=1)
            ]


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

.. figure:: oneof/output.png

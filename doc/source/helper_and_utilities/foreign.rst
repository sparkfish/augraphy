*******************
ForeignAugmentation
*******************

.. autoclass:: augraphy.utilities.foreign.ForeignAugmentation
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
ForeignAugmentation allows the usage of imgaug or Albumentations function in Augraphy pipeline.

-------
Example
-------
In this example, ForeignAugmentation allows the usage of Rain augmenter from imgaug in ink phase of Augraphy pipeline.
Paper phase is using ColorPaper from Augraphy while ForeignAugmentation in post phase to allow the usage of ElasticTransform from Albumentations.::


    # import libraries
    from augraphy import *
    import cv2
    import numpy as np
    from augraphy.utilities.foreign import ForeignAugmentation

    # import imgaug and albumentations
    import imgaug.augmenters as iaa
    from albumentations.augmentations.geometric.transforms import ElasticTransform


    # initialize phases and pipeline
    # ink phase with imgaug
    ink_phase   = [ForeignAugmentation(foreignAugmentation=iaa.Rain(drop_size=(0.10, 0.20)),p=1)]
    # paper phase with augraphy
    paper_phase = [ColorPaper(hue_range=(0,10), saturation_range=(128, 255), p=1)]
    # post phase with albumentations
    post_phase  = [ForeignAugmentation(foreignAugmentation=ElasticTransform(alpha_affine=100, p=1),p=1)]

    # initialize pipeline
    pipeline    = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    # create input image
    image = np.full((1200, 1200,3), 128, dtype="uint8")
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

.. figure:: foreign/output.png

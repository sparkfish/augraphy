************
SectionShift
************

.. autoclass:: augraphy.augmentations.sectionshift.SectionShift
    :members:
    :undoc-members:
    :show-inheritance:

--------
Overview
--------
The SectionShift augmentation shifts single or multiple sections of image in horizontal, vertical or both directions, creating an effect of shifted image sections. For additional input such as mask and keypoints, they are fully supported. For bounding boxes, they will be affected only if the starting point or ending point of the box in the shifting area.

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
In this example, a SectionShift augmentation instance is initialized and the number of shifting sections is set to 5 (5,5).
The shifting location is set of random coordinates ("random").
The shifting value in both horizontal and vertical direction is set to 20 (20,20).
The new value in the shifted area is filled with white color (255,255,255).

Code example:

::

    sectionshift = SectionShift(section_shift_number_range = (5,5),
                                section_shift_locations = "random",
                                section_shift_x_range = (20,20),
                                section_shift_y_range = (20,20),
                                section_shift_fill_value = (255,255,255)
                                )

    img_sectionshift = sectionshift(image)
    cv2.imshow("sectionshift", img_sectionshift)


Augmented image:

.. figure:: augmentations/sectionshift/section_shift.png


---------
Example 2
---------
In this example, a SectionShift augmentation will be applied to additional inputs such as mask, keypoints and bounding boxes.
The SectionShift augmentation will be using the default parameters value and hence no additional parameters value will be specified.

Code example:

::

    sectionshift = SectionShift()

    img_sectionshift, mask, keypoints, bounding_boxes = sectionshift(image=image, mask=mask, keypoints=keypoints, bounding_boxes=bounding_boxes)

    cv2.imshow("sectionshift", img_sectionshift)


Input mask:

.. figure:: augmentations/input_mask.png

Input keypoints:

.. figure:: augmentations/input_keypoints.png

Input bounding boxes:

.. figure:: augmentations/input_bounding_boxes.png

Augmented image:

.. figure:: augmentations/sectionshift/section_shift2.png

Augmented mask:

.. figure:: augmentations/sectionshift/section_shift2_mask.png

Augmented keypoints:

.. figure:: augmentations/sectionshift/section_shift2_keypoints.png

Augmented bounding boxes:

.. figure:: augmentations/sectionshift/section_shift2_bounding_boxes.png

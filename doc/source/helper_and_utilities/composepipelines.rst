****************
ComposePipelines
****************

.. autoclass:: augraphy.utilities.composepipelines.ComposePipelines
    :members:
    :undoc-members:
    :show-inheritance:


--------
Overview
--------
ComposePipelines allows the composition of multiple AugraphyPipelines. The consecutive pipelines will be applied on the output of prior pipeline. The output of ComposePipelines is a dict and the name of final output will be in the format of "pipeline(n-1)-output", where n is the number of total pipelines.

-------
Example
-------
In this example, ComposePipelines is use to compose three different pipelines.
::

    # import libraries
    from augraphy import *
    import cv2
    import numpy as np
    from augraphy.utilities.composepipelines import ComposePipelines

    # initialize pipeline1
    ink_phase1   = [InkBleed(p=1)]
    paper_phase1 = [DirtyRollers(p=1)]
    post_phase1  = [WaterMark(p=1)]
    pipeline1    = AugraphyPipeline(ink_phase1, paper_phase1, post_phase1)

    # initialize pipeline2
    ink_phase2   = [BleedThrough(p=1)]
    paper_phase2 = [DirtyDrum(p=1)]
    post_phase2  = [Faxify(p=1)]
    pipeline2    = AugraphyPipeline(ink_phase2, paper_phase2, post_phase2)

    # initialize pipeline3
    ink_phase3   = [Letterpress(p=1)]
    paper_phase3 = [BadPhotoCopy(p=1)]
    post_phase3  = [Folding(p=1)]
    pipeline3    = AugraphyPipeline(ink_phase3, paper_phase3, post_phase3)

    # compose pipelines
    compose_pipeline = ComposePipelines([pipeline1, pipeline2, pipeline3])

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
    augmented_image = compose_pipeline(image)["pipeline2-output"]


Input image:

.. figure:: input/input.png

Augmented image:

.. figure:: composepipelines/output.png

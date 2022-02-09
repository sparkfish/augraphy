import random

import cv2

from augraphy import *


def harley_davidson(clean):

    ink = [
        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(3, 5),
            markup_type="strikethrough",
            markup_color=(0, 0, 0),
            single_word_mode=True,
            repetitions=1,
        ),
        Faxify(
            monochrome=0,
            invert=1,
            half_kernel_size=1,
            angle=45,
            sigma=2,
        ),
        Faxify(
            monochrome=1,
            monochrome_method="Otsu",
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(4, 8),
            noise_size=(1, 4),
            noise_sparsity=(1, 1),
            noise_concentration=(1, 1),
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 5),
            kernel_size=(3, 3),
            severity=(0.1, 0.1),
        ),
    ]

    paper = []
    post = []

    pipeline = AugraphyPipeline(ink, paper, post)

    return pipeline.augment(clean)["output"]

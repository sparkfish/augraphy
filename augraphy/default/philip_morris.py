import random

import cv2

from augraphy import *


def philip_morris(clean):
    ink = [
        Faxify(
            monochrome=1,
            monochrome_method="Otsu",
        ),
        InkBleed(
            intensity_range=(0.9, 0.9),
            color_range=(0, 0),
            kernel_size=(3, 3),
            severity=(0.9, 0.9),
        ),
        BadPhotoCopy(
            noise_type=3,
            noise_iteration=(5, 8),
            noise_size=(1, 4),
            noise_sparsity=(0.025, 0.025),
            noise_concentration=(0.025, 0.025),
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(4, 8),
            noise_size=(1, 4),
            noise_sparsity=(1, 1),
            noise_concentration=(0.5, 0.8),
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 0),
            kernel_size=(3, 3),
            severity=(0.9, 0.9),
        ),
    ]
    paper = []
    post = []

    pipeline = AugraphyPipeline(ink, paper, post)

    return pipeline.augment(clean)["output"]

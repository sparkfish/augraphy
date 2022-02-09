import random

import cv2
import numpy as np

from augraphy import *


def marlboro_500(clean):
    # manual cropping and add noise

    # clean[250:, :] = clean[:-250, :]
    # clean[:250, :] = 255
    # clean[2150, 1400:1555] = 0

    # create augmentation pipeline

    ink = [
        Geometric(scale=(2, 2)),
        Faxify(
            monochrome=0,
            invert=1,
            half_kernel_size=1,
            angle=35,
            sigma=2,
        ),
        Geometric(scale=(0.5, 0.5)),
        Faxify(
            monochrome=1,
            monochrome_method="Adaptive",
            adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(4, 8),
            noise_size=(1, 4),
            noise_sparsity=(1, 1),
            noise_concentration=(0.5, 0.8),
        ),
        Letterpress(
            n_samples=(2800, 2800),
            n_clusters=(600, 680),
            std_range=(1500, 1500),
            value_range=(245, 255),
            value_threshold_range=(128, 128),
            blur=0,
        ),
    ]

    paper = []
    post = []

    pipeline = AugraphyPipeline(ink, paper, post)

    return pipeline.augment(clean)["output"]

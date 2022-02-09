import random

import cv2

from augraphy import *


def heparin_testosterone_rats(clean):
    # create augmentation pipeline

    ink = [
        PageBorder(
            side="bottom",
            pages=40,
            noise_intensity_range=(0.5, 0.8),
            width_range=(30, 30),
        ),
        Geometric(rotate_range=(-1, -1)),
        Geometric(crop=(0, 0, -1, 2215)),
        Faxify(
            monochrome=1,
            monochrome_method="Simple",
            monochrome_threshold=190,
        ),
        Letterpress(
            n_samples=(300, 300),
            n_clusters=(25000, 25000),
            std_range=(80, 80),
            value_range=(255, 255),
            value_threshold_range=(255, 255),
            blur=0,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(4, 8),
            noise_size=(1, 4),
            noise_sparsity=(1, 1),
            noise_concentration=(0.8, 0.9),
        ),
    ]

    paper = []
    post = [
        SubtleNoise(range=1),
        Jpeg(p=1),
    ]

    pipeline = AugraphyPipeline(ink, paper, post)

    return pipeline.augment(clean)["output"]

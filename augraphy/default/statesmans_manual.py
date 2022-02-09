import random

import cv2
import numpy as np

from augraphy import *


def statesmans_manual(clean):
    # create mask of noise

    blank_image = np.full_like(clean, fill_value=255).astype("uint8")

    # object for badphotocopy effect 1
    badphotocopy1 = BadPhotoCopy(
        noise_type=3,
        noise_value=(50, 70),
        noise_sparsity=(0.05, 0.05),
        noise_concentration=(0.1, 0.1),
        blur_noise=1,
        blur_noise_kernel=(31, 31),
    )

    # object for badphotocopy effect 2
    badphotocopy2 = BadPhotoCopy(
        noise_type=3,
        noise_value=(150, 160),
        noise_sparsity=(0.4, 0.4),
        noise_concentration=(0.3, 0.3),
        blur_noise=1,
        blur_noise_kernel=(31, 31),
    )

    noise_mask1 = badphotocopy1(blank_image)
    noise_mask2 = badphotocopy2(blank_image)

    # stack both masks by using min of both masks
    noise_mask1[noise_mask1 > noise_mask2] = noise_mask2[noise_mask1 > noise_mask2]

    # create augmenation pipeline

    ink = [
        BadPhotoCopy(mask=noise_mask1, blur_noise=1, blur_noise_kernel=(31, 31)),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 5),
            kernel_size=(3, 3),
            severity=(0.1, 0.2),
        ),
        BleedThrough(
            intensity_range=(0.1, 0.2),
            color_range=(0, 224),
            ksize=(17, 17),
            sigmaX=0,
            alpha=0.3,
            offsets=(10, 20),
        ),
        Geometric(scale=(1, 1), rotate_range=(2, 2)),
    ]

    paper = []
    post = [
        SubtleNoise(p=1),
        Jpeg(p=1),
    ]

    pipeline = AugraphyPipeline(ink, paper, post)

    return pipeline.augment(clean)["output"]

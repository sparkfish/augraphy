import random

import cv2
import numpy as np

from augraphy import *


def aktives_und_passives_rauchen(clean):
    ink1 = [
        Geometric(crop=(150, 0, -1, -1)),
        BookBinding(
            radius_range=(10, 10),
            curve_intensity_range=(70, 70),
            mirror=1,
            mirror_range=(0.2, 0.2),
        ),
        Folding(
            fold_x=50,
            fold_deviation=(0, 0),
            fold_count=1,
            fold_noise=1,
            gradient_width=(0.18, 0.18),
            gradient_height=(0.02, 0.02),
        ),
        Geometric(crop=(100, 0, -1, 2100)),
        Faxify(
            monochrome=1,
            monochrome_method="Otsu",
        ),
    ]

    paper1 = []
    post1 = []
    pipeline1 = AugraphyPipeline(ink1, paper1, post1)

    img_output1 = pipeline1.augment(clean)["output"]

    img_blank = np.full((2500, 1900), fill_value=255).astype("uint8")
    img_blank[:200, :1760] = 0

    ink2 = [
        BindingsAndFasteners(
            foreground=img_output1,
            overlay_types="darken",
            ntimes=1,
            nscales=(1, 1),
            edge="left",
            edge_offset=0,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(4, 8),
            noise_size=(1, 4),
            noise_sparsity=(1, 1),
            noise_concentration=(0.5, 0.8),
        ),
    ]

    paper2 = []
    post2 = []

    pipeline2 = AugraphyPipeline(ink2, paper2, post2)
    return pipeline2.augment(img_blank)["output"]

import random

import cv2
import numpy as np

from augraphy import *


def lorillard(clean):
    # create noise mask, subject to change later

    ysize, xsize = clean.shape[:2]

    # blank image, each to create different kind of noise masks
    blank_mask = np.full((int(clean.shape[0]), int(clean.shape[1])), fill_value=255).astype("uint8")
    blank_mask11 = np.full((int(clean.shape[0] / 8), int(clean.shape[1] / 8)), fill_value=255).astype("uint8")
    blank_mask12 = np.full((int(clean.shape[0] / 4), int(clean.shape[1] / 4)), fill_value=255).astype("uint8")
    blank_mask13 = np.full((int(clean.shape[0] / 2), int(clean.shape[1] / 2)), fill_value=255).astype("uint8")
    blank_mask2 = np.full((int(clean.shape[0] / 8), int(clean.shape[1] / 8)), fill_value=255).astype("uint8")
    blank_mask3 = np.full((int(clean.shape[0] / 8), int(clean.shape[1] / 8)), fill_value=255).astype("uint8")

    # create object for each noise effect
    badphotocopy11 = BadPhotoCopy(
        noise_type=6,
        noise_value=(0, 0),
        noise_sparsity=(0.4, 0.4),
        noise_concentration=(0.5, 0.5),
    )

    badphotocopy12 = BadPhotoCopy(
        noise_type=6,
        noise_value=(0, 0),
        noise_sparsity=(0.3, 0.3),
        noise_concentration=(0.5, 0.5),
    )

    badphotocopy13 = BadPhotoCopy(
        noise_type=6,
        noise_value=(0, 0),
        noise_sparsity=(0.2, 0.2),
        noise_concentration=(0.5, 0.5),
    )

    badphotocopy2 = BadPhotoCopy(
        noise_type=5,
        noise_value=(0, 0),
        noise_sparsity=(0.1, 0.1),
        noise_concentration=(0.1, 0.1),
    )

    badphotocopy3 = BadPhotoCopy(
        noise_type=6,
        noise_value=(0, 0),
        noise_sparsity=(0.1, 0.1),
        noise_concentration=(0.15, 0.15),
    )

    # get noise mask
    badphotocopy_mask11 = 255 - badphotocopy11(blank_mask11)
    badphotocopy_mask12 = 255 - badphotocopy11(blank_mask12)
    badphotocopy_mask13 = 255 - badphotocopy11(blank_mask13)
    badphotocopy_mask2 = 255 - badphotocopy2(blank_mask2)
    badphotocopy_mask3 = 255 - badphotocopy3(blank_mask3)

    # resize each noise mask back to input size
    badphotocopy_mask11 = resized = cv2.resize(badphotocopy_mask11, (xsize, ysize), interpolation=cv2.INTER_AREA)
    badphotocopy_mask12 = resized = cv2.resize(badphotocopy_mask12, (xsize, ysize), interpolation=cv2.INTER_AREA)
    badphotocopy_mask13 = resized = cv2.resize(badphotocopy_mask13, (xsize, ysize), interpolation=cv2.INTER_AREA)
    badphotocopy_mask2 = resized = cv2.resize(badphotocopy_mask2, (xsize, ysize), interpolation=cv2.INTER_AREA)
    badphotocopy_mask3 = resized = cv2.resize(badphotocopy_mask3, (xsize, ysize), interpolation=cv2.INTER_AREA)

    # combined all noise mask
    badphotocopy_mask = badphotocopy_mask11 + badphotocopy_mask12 + badphotocopy_mask13 + badphotocopy_mask2

    # create dirty drum noise mask object
    dirty_drum1 = DirtyDrum(
        line_width_range=(18, 18),
        direction=1,
        noise_intensity=0.95,
        ksize=(3, 3),
        sigmaX=0,
    )

    # create letterpress noise mask object
    letter_press1 = Letterpress(
        n_samples=(5000, 6000),
        n_clusters=(120, 120),
        std_range=(12000, 15000),
        value_range=(255, 255),
    )

    # create letterpress noise mask
    letter_press_mask = letter_press1(255 - blank_mask)

    # create letterpress noise mask
    dirty_drum_mask = 255 - dirty_drum1(blank_mask)
    dirty_drum_mask[dirty_drum_mask > 0] = 1
    dirty_drum_mask[letter_press_mask < 5] = 0
    dirty_drum_mask[letter_press_mask > 5] = 1

    # combine letterpress, dirty and badphotocopy noise mask
    dirty_mask = (255 - (dirty_drum_mask * badphotocopy_mask)).astype("int")
    dirty_mask = 255 - ((255 - dirty_mask) + badphotocopy_mask3)
    dirty_mask -= 200
    dirty_mask[dirty_mask < 0] = 0
    dirty_mask = dirty_mask.astype("uint8")

    # pipeline 1

    ink1 = [
        Faxify(
            monochrome=1,
            monochrome_method="Otsu",
        ),
        PageBorder(side="bottom", noise_intensity_range=(0.5, 0.8), width_range=(4, 5)),
        # PageBorder(side="right", noise_intensity_range=(0.5 , 0.8),width_range=(4,5)),
        BadPhotoCopy(mask=dirty_mask, noise_type=5),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 5),
            kernel_size=(3, 3),
            severity=(0.4, 0.5),
        ),
        Geometric(rotate_range=(-3, -3)),
    ]

    paper1 = []
    post1 = [
        SubtleNoise(p=1),
        Jpeg(p=1),
    ]

    # create pipeline 1 and applies augmentation
    pipeline1 = AugraphyPipeline(ink1, paper1, post1)
    img_output1 = pipeline1.augment(clean)["output"]

    # pipeline 2

    img_blank = np.full_like(img_output1, fill_value=255).astype("uint8")

    ink2 = [
        BindingsAndFasteners(
            foreground=img_output1,
            overlay_types="darken",
            ntimes=1,
            nscales=(1, 1),
            edge="bottom",
            edge_offset=150,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            effect_type="clips",
            ntimes=2,
            nscales=(8, 8),
            edge="bottom",
            edge_offset=40,
        ),
        PageBorder(
            side="left",
            width_range=(6, 7),
            pages=5,
            noise_intensity_range=(0.0, 0.2),
        ),
        PageBorder(
            side="bottom",
            width_range=(8, 12),
            pages=5,
            noise_intensity_range=(0.0, 0.2),
        ),
    ]

    paper2 = []
    post2 = []

    # create pipeline 2 and applies augmentation
    pipeline2 = AugraphyPipeline(ink2, paper2, post2)

    return pipeline2.augment(img_blank)["output"]

import random
import shutil

import cv2
import numpy as np

from augraphy import *
from augraphy.utilities.figsharedownloader import FigshareDownloader


def let_us_have_facts(clean):

    fsdl = FigshareDownloader()
    fsdl.download_file_by_id(33912014, "binderclip.jpg")
    fsdl.download_file_by_id(33912008, "binderring.jpg")
    binder_clip = cv2.imread("figshare/binderclip.jpg")
    binder_ring = cv2.imread("figshare/binderring.jpg")

    ink1 = [
        Markup(num_lines_range=(3, 3), markup_thickness_range=(2, 3), markup_color=(0, 0, 0)),
        BindingsAndFasteners(
            foreground=None,
            effect_type="punch_holes",
            ntimes=3,
            nscales=(1, 1),
            edge="left",
            edge_offset=20,
        ),
        PageBorder(side="right", noise_intensity_range=(0.5, 0.8), width_range=(5, 10)),
        PageBorder(side="top", noise_intensity_range=(0.5, 0.8), width_range=(5, 10)),
        PageBorder(side="left", noise_intensity_range=(0.5, 0.8), width_range=(5, 10)),
        Geometric(scale=(0.84, 0.85), rotate_range=(2, 2)),
    ]
    paper1 = []
    post1 = []

    pipeline1 = AugraphyPipeline(ink1, paper1, post1)
    img_output1 = pipeline1.augment(clean)["output"]

    # blank image to create pre-defined mask of noise
    blank_mask = np.full((clean.shape[0], clean.shape[1]), fill_value=255).astype("uint8")

    # object for badphotocopy with noise on top
    badphotocopy1 = BadPhotoCopy(
        noise_type=7,
        noise_value=(0, 1),
        noise_sparsity=(0.5, 0.5),
        noise_concentration=(1, 1),
    )

    # dirty drum object to create horizontal direction effect
    dirty_drum1 = DirtyDrum(
        line_width_range=(4, 8),
        direction=0,
        noise_intensity=0.95,
        ksize=(3, 3),
        sigmaX=0,
    )

    # dirty drum object to create vertical direction effect
    dirty_drum2 = DirtyDrum(
        line_width_range=(100, 100),
        direction=1,
        noise_intensity=0.95,
        ksize=(3, 3),
        sigmaX=0,
    )

    # letter press object to create cluster based noise
    letter_press1 = Letterpress(
        n_samples=(10000, 12000),
        n_clusters=(90, 100),
        std_range=(12000, 15000),
        value_range=(200, 255),
    )

    # get letter press mask and invert image
    letter_press_mask = letter_press1(255 - blank_mask)

    # get each directional mask and invert image
    dirty_drum_mask_h = 255 - dirty_drum1(blank_mask)
    dirty_drum_mask_v = 255 - dirty_drum2(blank_mask)

    # combine vertical and horizontal direction dirty drum mask
    dirty_drum_mask_h[dirty_drum_mask_h > 0] = 1
    dirty_drum_mask_v[dirty_drum_mask_v > 0] = 1
    dirty_drum_mask = dirty_drum_mask_h + dirty_drum_mask_v
    dirty_drum_mask[dirty_drum_mask > 0] = 1

    # produce cluster based noise effect
    dirty_drum_mask[letter_press_mask > 0] = 1

    # create badphotocopy mask with noise on top
    badphotocopy_mask = 255 - badphotocopy1(blank_mask)

    # create the final mask of noise
    dirty_mask = 255 - (dirty_drum_mask * badphotocopy_mask)

    ink2 = [
        BindingsAndFasteners(
            foreground=img_output1,
            overlay_types="darken",
            ntimes=1,
            nscales=(1, 1),
            edge="top",
            edge_offset=600,
        ),
        BindingsAndFasteners(
            foreground=binder_clip,
            overlay_types="darken",
            ntimes=3,
            nscales=(2, 2),
            edge="bottom",
            edge_offset=30,
        ),
        BindingsAndFasteners(
            foreground=binder_ring,
            overlay_types="darken",
            ntimes=3,
            nscales=(2, 2),
            edge="right",
            edge_offset=10,
        ),
        Letterpress(
            n_samples=(400, 500),
            n_clusters=(500, 1000),
            std_range=(500, 500),
            value_range=(200, 255),
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_value=(0, 15),
            noise_sparsity=(0.01, 0.01),
            noise_concentration=(1, 1),
        ),
        BadPhotoCopy(mask=dirty_mask, noise_type=7),
        BadPhotoCopy(
            noise_type=7,
            noise_value=(0, 5),
            noise_sparsity=(0.001, 0.001),
            noise_concentration=(1, 1),
        ),
    ]

    paper2 = []
    post2 = []

    # create a blank image
    img_blank = np.full_like(clean, fill_value=255)

    # apply pipeline
    pipeline2 = AugraphyPipeline(ink2, paper2, post2)

    shutil.rmtree("figshare")

    return pipeline2.augment(img_blank)["output"]

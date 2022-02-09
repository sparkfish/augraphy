import random

import cv2
import numpy as np

from augraphy import *


def ira_jay_goldberg(clean):
    # create predefined foregrounds

    # get image size
    ysize, xsize = clean.shape[:2]

    # create top and bottom black bar
    img_black_box = np.full((200, xsize), fill_value=255).astype("uint8")
    img_black_box[:, : xsize - 250] = 0

    # create staple holes
    img_black_line = np.full((200, xsize), fill_value=255).astype("uint8")
    img_black_line[5:35, 275:279] = 0
    img_black_line[55:60, 273:276] = 0
    img_black_line[62:77, 274:277] = 0

    ink1 = [
        Faxify(
            monochrome=1,
            monochrome_method="Otsu",
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=img_black_box,
            ntimes=1,
            nscales=(1, 1),
            edge="top",
            edge_offset=0,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=img_black_line,
            ntimes=1,
            nscales=(1, 1),
            edge="top",
            edge_offset=280,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=img_black_box,
            ntimes=1,
            nscales=(1, 1),
            edge="bottom",
            edge_offset=100,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            effect_type="punch_holes",
            ntimes=3,
            nscales=(1.5, 1.5),
            edge="left",
            edge_offset=250,
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 5),
            kernel_size=(3, 3),
            severity=(0.2, 0.2),
        ),
    ]

    paper1 = []
    post1 = [
        SubtleNoise(p=1),
        Jpeg(p=1),
    ]

    pipeline = AugraphyPipeline(ink1, paper1, post1)

    return pipeline.augment(clean)["output"]

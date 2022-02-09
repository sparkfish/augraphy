import math
import random
import shutil

import cv2
import numpy as np

from augraphy import *
from augraphy.utilities.figsharedownloader import FigshareDownloader


def diahann_carroll(clean):

    fsdl = FigshareDownloader()
    fsdl.download_file_by_id(33912011, "1-holes.jpg")
    fsdl.download_file_by_id(33947567, "1-P-742.jpg")
    holes = cv2.imread("figshare/1-holes.jpg")
    p742 = cv2.imread("figshare/1-P-742.jpg")

    # create text with 600
    # create blank image
    img_600 = np.full((90, clean.shape[1]), fill_value=255).astype("uint8")

    # config for text
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    location = (1550, 70)
    fontScale = 2
    fontColor = (0, 0, 0)
    thickness = 2
    lineType = 2

    # draw text 600 into image
    img_600 = cv2.putText(
        img_600,
        "600",
        location,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )

    # create objects for rotation and letterpress effect
    rotate_object = Geometric(rotate_range=(45, 45))
    letterpress_object = Letterpress(
        n_samples=(1000, 1000),
        n_clusters=(1000, 1000),
        std_range=(800, 800),
        value_range=(255, 255),
        value_threshold_range=(150, 150),
        blur=0,
    )

    img_600 = letterpress_object(rotate_object(img_600))
    img_600 = img_600[:180, :]
    img_600 = cv2.resize(img_600, (clean.shape[1], 180), interpolation=cv2.INTER_AREA)

    # defining the pipeline
    ink = [
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=img_600,
            ntimes=1,
            nscales=(1, 1),
            edge="bottom",
            edge_offset=0,
        ),
        Geometric(
            scale=(0.5, 0.5),
            rotate_range=(0, 0),
        ),
        PageBorder(
            side="right",
            noise_intensity_range=(0.5, 0.8),
            width_range=(3, 3),
        ),
        Geometric(translation=(-50, 0)),
        Letterpress(
            n_samples=(1800, 1800),
            n_clusters=(1800, 1800),
            std_range=(1500, 1500),
            value_range=(220, 240),
            value_threshold_range=(15, 15),
            blur=0,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=holes,
            ntimes=1,
            nscales=(1.5, 1.5),
            edge="top",
            edge_offset=10,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=p742,
            ntimes=1,
            nscales=(1.5, 1.5),
            edge="bottom",
            edge_offset=10,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(1, 1),
            noise_value=(0, 15),
            noise_sparsity=(0.01, 0.01),
            noise_concentration=(1, 1),
        ),
        Geometric(rotate_range=(-2, -2)),
        LowInkRandomLines(
            count_range=(1200, 1500),
            use_consistent_lines=False,
        ),
    ]

    paper = []
    post = []

    pipeline = AugraphyPipeline(ink, paper, post)

    shutil.rmtree("figshare")

    return pipeline.augment(clean)["output"]

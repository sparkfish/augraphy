"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline(img):

    ink_phase = [
        Dithering(p=0.5),
        InkBleed(p=0.5),
        BleedThrough(p=0.5),
        Letterpress(p=0.5),
        OneOf(
            [
                LowInkRandomLines(use_consistent_lines=False),
                LowInkRandomLines(use_consistent_lines=True),
                LowInkPeriodicLines(use_consistent_lines=False),
                LowInkPeriodicLines(use_consistent_lines=True),
            ],
        ),
    ]

    paper_phase = [
        PaperFactory(p=0.5),
        OneOf(
            [
                AugmentationSequence(
                    [
                        NoiseTexturize(),
                        BrightnessTexturize(),
                    ],
                ),
                AugmentationSequence(
                    [
                        BrightnessTexturize(),
                        NoiseTexturize(),
                    ],
                ),
            ],
            p=0.5,
        ),
        Brightness(p=0.5),
    ]

    post_phase = [
        OneOf([PageBorder(), DirtyRollers()], p=0.5),
        OneOf([LightingGradient(), Brightness()], p=0.5),
        DirtyDrum(p=0.5),
        SubtleNoise(p=0.5),
        Jpeg(p=0.5),
        Folding(p=0.5),
        Markup(p=0.5),
        PencilScribbles(p=0.5),
        OneOf([BookBinding(), BindingsAndFasteners()], p=0.5),
        BadPhotoCopy(p=0.5),
        Gamma(p=0.5),
        Geometric(p=0.5),
        Faxify(p=0.5),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase, log=False)

    return pipeline.augment(img)

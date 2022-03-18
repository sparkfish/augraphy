"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline(img):

    ink_phase = [
        #        Dithering(dither="ordered", order=11, p=1),
        #        InkBleed(p=0.5),
        #        BleedThrough(p=1),
        #        BindingsAndFasteners(p=1),
        #        Letterpress(p=0.5),
        #        OneOf(
        #            [
        #                LowInkRandomLines(use_consistent_lines=False),
        #                LowInkRandomLines(use_consistent_lines=True),
        #                LowInkPeriodicLines(use_consistent_lines=False),
        #                LowInkPeriodicLines(use_consistent_lines=True),
        #            ],
        #        ),
    ]

    paper_phase = [
        #                PaperFactory(tile_texture_shape = (250, 250),
        #                             texture_path ="./paper_textures",
        #                             p=1),
        #        OneOf(
        #            [
        #                AugmentationSequence(
        #                    [
        #                        NoiseTexturize(),
        #                        BrightnessTexturize(),
        #                    ],
        #                ),
        #                AugmentationSequence(
        #                    [
        #                        BrightnessTexturize(),
        #                        NoiseTexturize(),
        #                    ],
        #                ),
        #            ],
        #            p=0.5,
        #        ),
        Brightness(p=0.5),
    ]

    post_phase = [
        #        LightingGradient(p=1),
        #        OneOf([PageBorder(), DirtyRollers()], p=0.5),
        #                OneOf([LightingGradient(), Brightness()], p=0.5),
        #                        DirtyDrum(p=1),
        #                SubtleNoise(p=1),
        #                Jpeg(p=0.5),
        #                Folding(p=0.5),
        #        Markup(p=1),
        #                PencilScribbles(p=0.5),
        #                OneOf([BookBinding(), BindingsAndFasteners()], p=0.5),
        #        BadPhotoCopy(p=1),
        #                BleedThrough(p=1),
        #                Gamma(p=0.5),
        #                Geometric(p=0.5),
        #        Faxify(p=1),
        LightingGradient(p=1),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase, log=True)

    return pipeline.augment(img)

"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline():
    ink_phase = AugmentationSequence(
        [
            Dithering(layer="ink"),
            InkBleed(layer="ink"),
            BleedThrough(layer="ink"),
            Letterpress(layer="ink"),
            OneOf(
                [
                    LowInkRandomLines(layer="ink", use_consistent_lines=False),
                    LowInkRandomLines(layer="ink", use_consistent_lines=True),
                    LowInkPeriodicLines(layer="ink", use_consistent_lines=False),
                    LowInkPeriodicLines(layer="ink", use_consistent_lines=True),
                ],
            ),
            GaussianBlur(layer="ink"),
        ],
    )

    paper_phase = AugmentationSequence(
        [
            PaperFactory(),
            OneOf(
                [
                    AugmentationSequence(
                        [
                            NoiseTexturize(layer="paper"),
                            BrightnessTexturize(layer="paper"),
                            GaussianBlur("paper", [(3, 3), (3, 5), (5, 3), (5, 5)]),
                        ],
                    ),
                    AugmentationSequence(
                        [
                            BrightnessTexturize(layer="paper"),
                            NoiseTexturize(layer="paper"),
                            GaussianBlur("paper", [(3, 3), (3, 5), (5, 3), (5, 5)]),
                        ],
                    ),
                ],
            ),
            Brightness("paper"),
        ],
    )

    post_phase = AugmentationSequence(
        [
            OneOf([PageBorder(layer="post"), DirtyRollers(layer="post")]),
            OneOf([LightingGradient(), Brightness(layer="post")]),
            DirtyDrum(layer="post"),
            SubtleNoise(layer="post"),
            Jpeg(layer="post"),
            Folding(layer="post"),
        ],
    )

    return AugraphyPipeline(ink_phase, paper_phase, post_phase)

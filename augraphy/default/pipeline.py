"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline():
    ink_phase = AugmentationSequence(
        [
            InkBleed(),
            BleedThrough(),
            Letterpress(),
            OneOf(
                [
                    LowInkRandomLines(use_consistent_lines=False),
                    LowInkRandomLines(use_consistent_lines=True),
                    LowInkPeriodicLines(use_consistent_lines=False),
                    LowInkPeriodicLines(use_consistent_lines=True),
                ]
            ),
            GaussianBlur("ink"),
        ]
    )

    paper_phase = AugmentationSequence(
        [
            PaperFactory(),
            OneOf(
                [
                    AugmentationSequence(
                        [
                            NoiseTexturize(),
                            BrightnessTexturize(),
                            GaussianBlur("paper", [(3, 3), (3, 5), (5, 3), (5, 5)]),
                        ]
                    ),
                    AugmentationSequence(
                        [
                            BrightnessTexturize(),
                            NoiseTexturize(),
                            GaussianBlur("paper", [(3, 3), (3, 5), (5, 3), (5, 5)]),
                        ]
                    ),
                ]
            ),
            Brightness("paper"),
        ]
    )

    post_phase = AugmentationSequence(
        [
            OneOf([PageBorder(), DirtyRollers()]),
            OneOf([LightingGradient(), Brightness("post")]),
            DirtyDrum(),
            SubtleNoise(),
            Jpeg(),
            Folding(),
        ]
    )

    return AugraphyPipeline(ink_phase, paper_phase, post_phase)

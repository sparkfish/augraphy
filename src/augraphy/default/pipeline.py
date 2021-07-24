"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *

def default_augraphy_pipeline():
    ink_phase = AugmentationSequence([
        InkBleedAugmentation(),
        BleedThroughAugmentation(),
	LetterpressAugmentation(),
        FoldingAugmentation(),
        OneOf([
            LowInkRandomLinesAugmentation(use_consistent_lines=False),
            LowInkRandomLinesAugmentation(use_consistent_lines=True),
            LowInkPeriodicLinesAugmentation(use_consistent_lines=False),
            LowInkPeriodicLinesAugmentation(use_consistent_lines=True),
        ]),
        GaussianBlurAugmentation('ink')
    ])

    paper_phase = AugmentationSequence([
        PaperFactory(),
        OneOf([
            AugmentationSequence([
                NoiseTexturizeAugmentation(),
                BrightnessTexturizeAugmentation(),
                GaussianBlurAugmentation('paper', [(3,3), (3,5), (5,3), (5,5)]),
            ]),
            AugmentationSequence([
                BrightnessTexturizeAugmentation(),
                NoiseTexturizeAugmentation(),
                GaussianBlurAugmentation('paper', [(3,3), (3,5), (5,3), (5,5)]),
            ])]),
        BrightnessAugmentation('paper')
    ])


    post_phase = AugmentationSequence([
        DirtyRollersAugmentation(),
        OneOf([
            LightingGradientAugmentation(),
            BrightnessAugmentation('post')
        ]),
        DirtyDrumAugmentation(),
        SubtleNoiseAugmentation(),
        JpegAugmentation()
    ])

    return AugraphyPipeline(ink_phase, paper_phase, post_phase)

from Augraphy.AugraphyPipeline import AugraphyPipeline
from Augraphy.base import AugmentationSequence, OneOf
from Augraphy.augmentations import *
from Augraphy.paperfactory import PaperFactory

__all__ = [AugraphyPipeline]


def default_augraphy_pipeline():
    ink_phase = AugmentationSequence(
        [
            InkBleedAugmentation(),
            DustyInkAugmentation(),
            LowInkBlobsAugmentation(),
            OneOf(
                [
                    LowInkRandomLinesAugmentation(use_consistent_lines=False),
                    LowInkRandomLinesAugmentation(use_consistent_lines=True),
                    LowInkPeriodicLinesAugmentation(use_consistent_lines=False),
                    LowInkPeriodicLinesAugmentation(use_consistent_lines=True),
                ]
            ),
            GaussianBlurAugmentation("ink", probability=1),
        ]
    )

    paper_phase = AugmentationSequence(
        [
            PaperFactory(),
            OneOf(
                [
                    AugmentationSequence(
                        [
                            NoiseTexturizeAugmentation(probability=1.0),
                            BrightnessTexturizeAugmentation(),
                            GaussianBlurAugmentation(
                                "paper", [(3, 3), (3, 5), (5, 3), (5, 5)]
                            ),
                        ]
                    ),
                    AugmentationSequence(
                        [
                            BrightnessTexturizeAugmentation(probability=1.0),
                            NoiseTexturizeAugmentation(),
                            GaussianBlurAugmentation(
                                "paper", [(3, 3), (3, 5), (5, 3), (5, 5)]
                            ),
                        ]
                    ),
                ]
            ),
            BrightnessAugmentation("paper"),
        ]
    )

    post_phase = AugmentationSequence(
        [
            DirtyRollersAugmentation(probability=0.0),
            OneOf([LightingGradientAugmentation(), BrightnessAugmentation("post")]),
            SubtleNoiseAugmentation(),
            JpegAugmentation(),
        ]
    )

    return AugraphyPipeline(ink_phase, paper_phase, post_phase)

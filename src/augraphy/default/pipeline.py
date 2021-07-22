"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy.base.paperfactory import PaperFactory
from augraphy.base.oneof import OneOf
from augraphy.base.augmentationsequence import AugmentationSequence
from augraphy.base.augmentationpipeline import AugraphyPipeline


from augraphy.augmentations.bleedthrough import BleedThroughAugmentation
from augraphy.augmentations.brightness import BrightnessAugmentation
from augraphy.augmentations.brightnesstexturize import BrightnessTexturizeAugmentation
from augraphy.augmentations.dirtydrum import DirtyDrumAugmentation
from augraphy.augmentations.dirtyrollers import DirtyRollersAugmentation
from augraphy.augmentations.folding import FoldingAugmentation
from augraphy.augmentations.gaussianblur import GaussianBlurAugmentation
from augraphy.augmentations.inkbleed import InkBleedAugmentation
from augraphy.augmentations.jpeg import JpegAugmentation
from augraphy.augmentations.letterpress import LetterpressAugmentation
from augraphy.augmentations.lightinggradient import LightingGradientAugmentation
from augraphy.augmentations.lowinkperiodiclines import LowInkPeriodicLinesAugmentation
from augraphy.augmentations.lowinkrandomlines import LowInkRandomLinesAugmentation
from augraphy.augmentations.noisetexturize import NoiseTexturizeAugmentation
from augraphy.augmentations.subtlenoise import SubtleNoiseAugmentation

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

    paper_phase = AugmentationSequence([PaperFactory(),
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


    post_phase = AugmentationSequence([ DirtyRollersAugmentation(),
                                        OneOf([
                                                LightingGradientAugmentation(),
                                                BrightnessAugmentation('post')
                                                ]),
                                        DirtyDrumAugmentation(),
                                        SubtleNoiseAugmentation(),
                                        JpegAugmentation()
                                        ])

    return AugraphyPipeline(ink_phase, paper_phase, post_phase)

import string
import random
import cv2
import numpy as np
import PIL
import os
from pdf417gen import encode, render_image

from Augraphy import AugraphyPipeline
from Augraphy.Augmentations import *

def create_pdf417():
    type = 'pdf417'
    codeword_multiplier = random.randint(1, 9)
    text_length = codeword_multiplier * 50
    text = ''.join(random.choices(string.ascii_letters + string.digits + "&,:#-.$/+%* =^;<>@[\\]_'~!|()?{}", k=text_length))
    scale = random.randint(2, 3)
    padding = 5
    columns = codeword_multiplier + 2
    ratio = random.randint(2, 4)
    encoded = encode(text, columns=columns)
    image = render_image(encoded, scale=scale, ratio=ratio, padding=padding)

    return image

img = np.array(create_pdf417())

ink_phase = AugmentationSequence([
                                  InkBleedAugmentation(),
                                  DustyInkAugmentation(),
                                  LowInkBlobsAugmentation(), 
                                  OneOf([
                                    LowInkRandomLinesAugmentation(count_range=(3, 10), use_consistent_lines=False), #line jitter, add range 
                                    LowInkRandomLinesAugmentation(count_range=(3, 10), use_consistent_lines=True), 
                                    LowInkPeriodicLinesAugmentation(count_range=(2, 2), period_range=(5, 5), use_consistent_lines=False), 
                                    LowInkPeriodicLinesAugmentation(count_range=(2, 2), period_range=(5, 5), use_consistent_lines=True), 
                                  ]),
                                  GaussianBlurAugmentation(probability=1)
                                ])

paper_phase = AugmentationSequence([
                                    OneOf([
                                      AugmentationSequence([
                                          NoiseTexturizeAugmentation(),
                                          BrightnessTexturizeAugmentation()
                                        ]),
                                      AugmentationSequence([
                                          BrightnessTexturizeAugmentation(probability=1.0),
                                          NoiseTexturizeAugmentation()
                                    ])]),
                                    GaussianBlurAugmentation([(3,3), (3,5), (5,3), (5,5)]),
                                    BrightnessAugmentation()
                            ])


post_phase = AugmentationSequence([DirtyRollersAugmentation(),
                                   OneOf([
                                        LightingGradientAugmentation(),
                                        BrightnessAugmentation()
                                      ]),
                                      JpegAugmentation(), 
                                      SubtleNoiseAugmentation()
                                    ])

pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

#img = cv2.imread("test.png")

for i in range(10):
    crappified = pipeline.augment(img)

    cv2.imshow("crappified", crappified)
    cv2.waitKey(1000)

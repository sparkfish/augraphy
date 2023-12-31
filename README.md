<p align="center">
    <img src="https://github.com/sparkfish/augraphy/blob/dev/images/logo/augraphy.png?raw=true" width="500" title="Augraphy Logo">
</p>

Augraphy is a Python library that creates multiple copies of original documents though an augmentation pipeline that randomly distorts each copy -- degrading the clean version into dirty and realistic copies rendered through synthetic paper printing, faxing, scanning and copy machine processes.

Highly-configurable pipelines apply adjustments to the originals to create realistic old or noisy documents by acting as a factory, producing almost an infinite number of variations from their source.  This simulation of realistic paper-oriented process distortions can create large amounts of training data for AI/ML processes to learn how to remove those distortions.

Treatments applied by Augraphy fabricate realistic documents that appear to have been printed on dirty laser or inkjet printers, scanned by dirty office scanners, faxed by low-resolution fax machines and otherwise mistreated by real-world paper handling office equipment.

# What makes Augraphy Magical?
Virtually no readily available datasets exist with both a _clean_ and _noisy_ version of target documents.  Augraphy addresses that problem by manufacturing large volumes of high-quality noisy documents to train alongside their clean source originals.

Training neural networks typically requires augmenting limited sources of data in a variety of ways so that networks can learn to generalize their solutions.  Networks designed to work with scanned document images must be trained with images that have the type of distortions and noise typical of real-world scanned office documents.

However, if we only have real-world dirty documents, then we don’t have a good way to know for sure what the right answer is when training a neural network.  By going in the reverse direction, starting with the clean document we hope a trained network will produce, we can simulate training data with dirty documents for which we already have a perfect original.

With flawless rendering of distorted "originals", we can train a model to undo all that distortion and restore the document to its original form.  It’s pretty much magic!

# How It Works
Augraphy's augmentation pipeline starts with an image of a clean document.  The pipeline begins by extracting the text and graphics from the source into an "ink" layer.  (Ink is synonymous with toner within Augraphy.)  The augmentation pipeline then distorts and degrades the ink layer.

A paper factory provides either a white page or a randomly-selected paper texture base.  Like the ink layer, the paper can also be processed through a pipeline to further provide random realistic paper textures.

After both the ink and paper phases are completed, processing continues by applying the ink, with its desired effects, to the paper.  This merged document image is then augmented further with distortions such as adding folds or other physical deformations or distortions that rely on simultaneous interactions of paper and ink layers.

The end result is an image that mimics real documents.

<p align="center" width="100%">
    <img src="https://github.com/sparkfish/augraphy/blob/dev/images/augraphy-pipeline.png?raw=true">
</p>

## Example Before / After Images
<p align="center" width="100%">
    <img src="https://github.com/sparkfish/augraphy/blob/dev/images/basic_examples.png?raw=true">
</p>


# Example Usage
To use the default pipeline which contains all available augmentations and sensible defaults:

```python
from augraphy import *

pipeline = default_augraphy_pipeline()

image = cv2.imread("image.png")

augmented = pipeline(image)

```

# Documentation
For full documentation, including installation and tutorials, check the [doc directory](https://github.com/sparkfish/augraphy/tree/dev/doc).

# List of Augmentations
## Pixel Level Augmentations
Pixel level augmentations apply augmentation to the input image only, that including alpha layer of the image. Additional inputs such as mask, keypoints or bounding boxes will not be affected.

|    Augmentation    |      Image      |   Alpha Layer   |
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |        ✓        |        -        |
|BindingsAndFasteners|        ✓        |        -        |
|BleedThrough        |        ✓        |        -        |
|Brightness          |        ✓        |        -        |
|BrightnessTexturize |        ✓        |        -        |
|ColorPaper          |        ✓        |        -        |
|ColorShift          |        ✓        |        -        |
|DelaunayTessellation|        ✓        |        -        |
|DirtyDrum           |        ✓        |        -        |
|DirtyRollers        |        ✓        |        -        |
|Dithering           |        ✓        |        -        |
|DotMatrix           |        ✓        |        -        |
|Faxify              |        ✓        |        -        |
|Gamma               |        ✓        |        -        |
|Hollow              |        ✓        |        -        |
|InkBleed            |        ✓        |        -        |
|InkColorSwap        |        ✓        |        -        |
|InkMottling         |        ✓        |        -        |
|Jpeg                |        ✓        |        -        |
|Letterpress         |        ✓        |        -        |
|LightingGradient    |        ✓        |        -        |
|LinesDegradation    |        ✓        |        -        |
|LowInkPeriodicLines |        ✓        |        -        |
|LowInkRandomLines   |        ✓        |        -        |
|LowLightNoise       |        ✓        |        -        |
|Markup              |        ✓        |        -        |
|NoiseTexturize      |        ✓        |        -        |
|NoisyLines          |        ✓        |        -        |
|PatternGenerator    |        ✓        |        -        |
|ReflectedLight      |        ✓        |        -        |
|Scribbles           |        ✓        |        -        |
|ShadowCast          |        ✓        |        -        |
|SubtleNoise         |        ✓        |        -        |
|VoronoiTessellation |        ✓        |        -        |
|WaterMark           |        ✓        |        -        |

## Spatial level Augmentations
Spatial level augmentations apply augmentation to all inputs such as image (including alpha layer), mask, keypoints and bounding boxes.


|    Augmentation    |      Image      |   Alpha Layer   |       Mask      |    Keypoints    | Bounding Boxes  |
|--------------------|----------------:|----------------:|----------------:|----------------:|----------------:|
|BookBinding         |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|Folding             |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|Geometric           |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|GlitchEffect        |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|InkShifter          |        ✓        |        ✓        |        ✓        |        ✕        |        ✕        |
|PageBorder          |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|SectionShift        |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |
|Squish              |        ✓        |        ✓        |        ✓        |        ✓        |        ✓*       |

Remarks: <br />
[-] : augmentation doesn't affect this input. <br />
[✓] : augmentation is supported on this input. <br />
[✕] : augmentation is not supported on this input. <br />
[✓*] : augmentation is supported on this input under certain criteria. <br />

# Benchmark Results
The benchmark results are computed with Augraphy 8.2 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz. The mask is using a binarized version of image. For keypoints, it is using 1000 random points in the image and for bounding boxes, 100 random bounding boxes with random size are used.

|    Augmentation    |Img/sec|Memory(MB)|Img/sec(mask)|Memory(MB)(mask)|Img/sec(keypoints)|Memory(MB)(keypoints)|Img/sec(bounding boxes)|Memory(MB)(bounding boxes)|
|--------------------|------:|---------:|------------:|---------------:|-----------------:|--------------------:|----------------------:|-------------------------:|
|BadPhotoCopy        |   0.17|    202.81|         0.12|          215.60|              0.13|               204.55|                   0.12|                    216.29|
|BindingsAndFasteners|  28.21|     21.08|        29.75|           21.02|             36.39|                21.02|                  29.09|                     21.23|
|BleedThrough        |   0.39|    684.69|         0.39|          684.69|              0.39|               684.69|                   0.39|                    684.69|
|BookBinding         |   0.09|    611.71|         0.08|          616.87|              0.09|               611.74|                   0.09|                    611.67|
|Brightness          |   4.92|    147.99|         4.90|          147.99|              4.95|               147.99|                   5.07|                    147.99|
|BrightnessTexturize |   1.83|    181.74|         1.83|          181.74|              1.84|               181.74|                   1.86|                    181.74|
|ColorPaper          |   4.83|    105.66|         4.99|          105.66|              4.96|               105.66|                   4.96|                    105.66|
|ColorShift          |   0.79|    126.94|         0.82|          126.94|              0.77|               126.94|                   0.76|                    126.94|
|DelaunayTessellation|   0.11|     60.41|         0.12|           60.29|              0.11|                60.36|                   0.10|                     60.37|
|DepthSimulatedBlur  |   0.01|     76.08|         0.01|           76.08|              0.01|                76.08|                   0.01|                     76.08|
|DirtyDrum           |   0.83|    482.51|         0.94|          481.56|              0.90|               481.68|                   0.92|                    481.52|
|DirtyRollers        |   1.47|    249.55|         1.77|          249.43|              1.80|               249.43|                   1.78|                    249.43|
|DirtyScreen         |   0.78|    435.36|         0.78|          435.36|              0.77|               435.36|                   0.77|                    435.36|
|Dithering           |   3.39|    126.82|         3.66|          126.80|              3.66|               126.81|                   3.79|                    126.80|
|DotMatrix           |   0.53|     80.75|         0.57|           80.52|              0.57|                80.52|                   0.57|                     80.52|
|DoubleExposure      |   1.64|     63.40|         1.67|           63.40|              1.62|                63.40|                   1.65|                     63.40|
|Faxify              |   1.37|    138.34|         1.43|          141.28|              1.41|               136.95|                   1.27|                    149.24|
|Folding             |   3.18|     57.50|         1.24|           60.40|              3.44|                57.60|                   3.55|                     57.20|
|Gamma               |  29.26|     25.36|        26.90|           25.36|             28.23|                25.36|                  32.03|                     25.36|
|Geometric           | 135.75|     12.68|       137.64|           12.68|            145.80|                12.68|                 127.10|                     12.68|
|GlitchEffect        |   1.14|    132.35|         1.10|          132.65|              1.03|               134.14|                   1.05|                    134.11|
|Hollow              |   0.17|    343.17|         0.17|          343.17|              0.17|               343.17|                   0.17|                    343.17|
|InkBleed            |   3.23|    177.51|         3.19|          177.51|              3.17|               177.51|                   3.16|                    177.51|
|InkColorSwap        |   3.47|     51.99|         3.58|           51.99|              3.51|                51.99|                   3.61|                     51.99|
|InkMottling         |   5.41|     55.99|         5.47|           55.99|              5.49|                55.99|                   5.39|                     55.99|
|InkShifter          |   0.17|    426.86|         0.15|          426.43|              0.17|               426.78|                   0.17|                    426.58|
|LCDScreenPattern    |   2.14|    494.09|         2.12|          493.62|              2.14|               494.74|                   2.13|                    496.46|
|Jpeg                |   5.55|     25.87|         5.60|           25.86|              5.52|                25.87|                   5.66|                     25.87|
|LensFlare           |   0.02|    405.97|         0.01|          405.82|              0.01|               405.82|                   0.01|                    405.82|
|Letterpress         |   0.35|    135.71|         0.33|          140.72|              0.34|               137.25|                   0.34|                    136.31|
|LightingGradient    |   0.37|    638.31|         0.38|          638.30|              0.39|               638.30|                   0.40|                    638.30|
|LinesDegradation    |   1.28|    174.76|         1.27|          174.93|              1.28|               174.92|                   1.31|                    174.59|
|LowInkPeriodicLines |   5.17|     12.75|         5.26|           12.75|              5.56|                12.75|                   5.10|                     12.75|
|LowInkRandomLines   |  91.52|     12.75|        86.12|           12.75|             87.58|                12.75|                  98.28|                     12.75|
|LowLightNoise       |   0.27|    481.95|         0.28|          481.95|              0.27|               481.95|                   0.27|                    481.95|
|Markup              |   2.33|    161.88|         2.41|          158.13|              2.58|               146.27|                   2.60|                    147.53|
|Moire               |   0.97|    575.74|         1.03|          575.57|              1.05|               575.57|                   1.05|                    575.57|
|NoiseTexturize      |   0.83|    249.36|         0.85|          249.36|              0.80|               249.36|                   0.82|                    249.36|
|NoisyLines          |   0.89|    446.65|         0.83|          448.43|              0.86|               447.88|                   0.85|                    448.52|
|PageBorder          |   0.49|    193.95|         0.49|          188.46|              0.48|               188.30|                   0.49|                    192.04|
|PatternGenerator    |   0.76|     51.53|         0.76|           51.50|              0.74|                51.50|                   0.74|                     51.50|
|ReflectedLight      |   0.06|    109.90|         0.06|          109.82|              0.06|               109.88|                   0.06|                    110.02|
|Scribbles           |   1.11|     94.73|         0.86|           96.90|              0.87|                96.96|                   0.86|                    100.55|
|SectionShift        | 117.15|     12.96|       101.71|           13.02|            107.88|                12.95|                 115.22|                     12.96|
|ShadowCast          |   0.75|     50.79|         0.68|           50.80|              0.68|                50.80|                   0.74|                     50.80|
|Squish              |   0.72|    450.44|         0.72|          450.79|              0.73|               450.83|                   0.76|                    451.00|
|Stains              |   1.11|    469.14|         1.14|          469.14|              1.16|               469.14|                   1.11|                    469.14|
|SubtleNoise         |   1.44|    215.55|         1.47|          215.55|              1.47|               215.55|                   1.48|                    215.55|
|VoronoiTessellation |   0.07|     58.07|         0.07|           57.74|              0.07|                57.89|                   0.07|                     58.13|
|WaterMark           |   2.09|    363.62|         1.86|          404.62|              1.78|               409.50|                   2.02|                    380.21|

# Alternative Augmentation Libraries
There are plenty of choices when it comes to [augmentation libraries](https://github.com/AgaMiko/data-augmentation-review).  However, only Augraphy is designed to address everyday office automation needs associated with paper-oriented process distortions that come from printing, faxing, scanning and copy machines.  Most other libraries focus on video and images pertinent to camera-oriented data sources and problem domains.  Augraphy is focused on supporting problems related to automation of document images such as OCR, form recognition, form data extraction, document classification, barcode decoding, denoising, document restoration, identity document data extraction, document cropping, etc.  Eventually, Augraphy will be able to support photo OCR problems with augmentations designed to emulate camera phone distortions.

# Contributing
Pull requests are _very_ welcome.  Please open an issue to propose and discuss feature requests and major changes.

# Citations
If you used Augraphy in your research, please cite the project.

BibTeX:
```
@inproceedings{augraphy_paper,
    author = {Groleau, Alexander and Chee, Kok Wei and Larson, Stefan and Maini, Samay and Boarman, Jonathan},
    title = {Augraphy: A Data Augmentation Library for Document Images},
    booktitle = {Proceedings of the 17th International Conference on Document Analysis and Recognition ({ICDAR})},
    year = {2023},
    url = {https://arxiv.org/pdf/2208.14558.pdf}
}

@software{augraphy_library,
    author = {The Augraphy Project},
    title = {Augraphy: an augmentation pipeline for rendering synthetic paper printing, faxing, scanning and copy machine processes},
    url = {https://github.com/sparkfish/augraphy},
    version = {8.2.6}
}
```

## Star History
Please add a "star" to the repo.  It's exciting to us when we see your interest, which keep us motivated to continue investing in the project!
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date" />
    <img src="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date" alt="Star History Chart" />
</picture>

# License
Copyright 2023 Sparkfish LLC

Augraphy is free and open-source software distributed under the terms of the [**MIT**](https://github.com/sparkfish/augraphy/blob/dev/LICENSE) license.

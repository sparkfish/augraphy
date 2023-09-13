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
    <img src="https://github.com/sparkfish/augraphy/blob/dev/images/Pipeline.png?raw=true">
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

augmented = pipeline(img)

```

# Documentation
For full documentation, including installation and tutorials, check the [doc directory](https://github.com/sparkfish/augraphy/tree/dev/doc).

# Benchmark Results
The benchmark results are computed with Augraphy 8.20 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.

|    Augmentation    |Images per second|Memory usage (MB)|
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |             0.37|           178.20|
|BindingsAndFasteners|            31.40|            21.43|
|BleedThrough        |             0.43|           684.69|
|BookBinding         |             0.11|           611.74|
|Brightness          |             5.75|           147.99|
|BrightnessTexturize |             2.16|           181.74|
|ColorPaper          |             5.07|           105.66|
|ColorShift          |             0.88|           114.26|
|DelaunayTessellation|             0.14|            60.00|
|DirtyDrum           |             0.96|           482.63|
|DirtyRollers        |             1.63|           249.52|
|Dithering           |             3.82|           126.82|
|DotMatrix           |             0.64|            80.77|
|Faxify              |             2.12|           126.74|
|Folding             |             2.62|            63.28|
|Gamma               |            34.39|            25.36|
|Geometric           |           183.89|            12.68|
|GlitchEffect        |             1.49|           126.94|
|Hollow              |             0.21|           343.17|
|InkBleed            |             4.00|           177.51|
|InkColorSwap        |             5.10|            51.99|
|InkMottling         |             7.60|            55.99|
|InkShifter          |             0.20|           426.89|
|Jpeg                |             6.38|            25.85|
|Letterpress         |             0.46|           158.10|
|LightingGradient    |             0.47|           638.31|
|LinesDegradation    |             1.51|           175.38|
|LowInkPeriodicLines |             6.19|            12.74|
|LowInkRandomLines   |           144.08|            12.74|
|LowLightNoise       |             0.32|           481.95|
|Markup              |             2.54|           154.12|
|NoiseTexturize      |             0.96|           249.36|
|NoisyLines          |             1.04|           446.89|
|PageBorder          |             0.56|           191.84|
|PatternGenerator    |             1.00|            51.53|
|ReflectedLight      |             0.08|           109.92|
|Scribbles           |             0.93|            99.10|
|SectionShift        |           154.45|            12.95|
|ShadowCast          |             0.86|            50.80|
|Squish              |             0.99|           443.70|
|SubtleNoise         |             1.82|           202.87|
|VoronoiTessellation |             0.09|            58.18|
|WaterMark           |             2.55|           390.55|

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
    version = {8.2.4}
}
```

## Star History
Please add a "star" to the repo.  It's exciting to us when we see your interest, which keep us motivated to continue investing in the project!
<a href="https://star-history.com/#Sparkfish/augraphy&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date" />
    <img src="https://api.star-history.com/svg?repos=Sparkfish/augraphy&type=Date" alt="Star History Chart" />
  </picture>
</a>

# License
Copyright 2023 Sparkfish LLC

Augraphy is free and open-source software distributed under the terms of the [**MIT**](https://github.com/sparkfish/augraphy/blob/dev/LICENSE) license.

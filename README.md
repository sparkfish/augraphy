<p align="center">
    <img src="https://github.com/sparkfish/augraphy/blob/dev/images/logo/augraphy.png?raw=true" width="600" title="Augraphy Logo">
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

img = cv2.imread("image.png")

data = pipeline.augment(img)

augmented = data["output"]
```

# Documentation
For full documentation, including installation and tutorials, check the [doc directory](https://github.com/sparkfish/augraphy/tree/dev/doc).

# Benchmark Results
The benchmark results are computed with Augraphy 8.20 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.

|    Augmentation    |Images per second|Memory usage (MB)|
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |             0.38|           178.20|
|BindingsAndFasteners|            29.34|            21.43|
|BleedThrough        |             0.39|           684.69|
|BookBinding         |             0.08|           611.74|
|Brightness          |             5.93|           147.99|
|BrightnessTexturize |             2.12|           181.74|
|ColorPaper          |             5.12|           105.66|
|ColorShift          |             0.87|           114.26|
|DelaunayTessellation|             0.14|            60.00|
|DirtyDrum           |             0.82|           482.63|
|DirtyRollers        |             1.58|           249.52|
|Dithering           |             3.96|           126.82|
|DotMatrix           |             0.22|            80.53|
|Faxify              |             2.02|           126.75|
|Folding             |             2.85|            63.27|
|Gamma               |            34.27|            25.36|
|Geometric           |           189.37|            12.68|
|GlitchEffect        |             1.48|           126.94|
|Hollow              |             0.21|           343.17|
|InkBleed            |             3.93|           177.51|
|InkColorSwap        |             5.22|            51.99|
|InkMottling         |             7.78|            55.99|
|InkShifter          |             0.17|           426.89|
|Jpeg                |             6.32|            25.85|
|Letterpress         |             0.45|           158.10|
|LightingGradient    |             0.44|           638.31|
|LinesDegradation    |             1.69|           175.38|
|LowInkPeriodicLines |             6.47|            12.74|
|LowInkRandomLines   |           114.02|            12.74|
|LowLightNoise       |             0.30|           481.95|
|Markup              |             2.97|           154.12|
|NoiseTexturize      |             0.73|           249.36|
|NoisyLines          |             0.86|           446.76|
|PageBorder          |             0.32|           201.22|
|PatternGenerator    |             1.68|            51.52|
|ReflectedLight      |             0.08|           109.97|
|Scribbles           |             1.00|            97.13|
|SectionShift        |           129.82|            12.96|
|ShadowCast          |             0.95|            50.79|
|Squish              |             0.26|           465.31|
|SubtleNoise         |             1.68|           202.87|
|VoronoiTessellation |             0.08|            57.48|
|WaterMark           |             2.70|           352.34|

# Alternative Augmentation Libraries
There are plenty of choices when it comes to [augmentation libraries](https://github.com/AgaMiko/data-augmentation-review).  However, only Augraphy is designed to address everyday office automation needs associated with paper-oriented process distortions that come from printing, faxing, scanning and copy machines.  Most other libraries focus on video and images pertinent to camera-oriented data sources and problem domains.  Augraphy is focused on supporting problems related to automation of document images such as OCR, form recognition, form data extraction, document classification, barcode decoding, denoising, document restoration, identity document data extraction, document cropping, etc.  Eventually, Augraphy will be able to support photo OCR problems with augmentations designed to emulate camera phone distortions.

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# Citations
If you used Augraphy in your research, please cite the project.

BibTeX:
```
@software{augraphy_library,
    author = {The Augraphy Project},
    title = {Augraphy: an augmentation pipeline for rendering synthetic paper printing, faxing, scanning and copy machine processes},
    url = {https://github.com/sparkfish/augraphy},
    version = {8.2.3}
}
```

# License
Copyright 2023 Sparkfish LLC

Augraphy is free and open-source software distributed under the terms of the [**MIT**](https://github.com/sparkfish/augraphy/blob/dev/LICENSE) license.

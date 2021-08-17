Augmentation pipeline for rendering printing, faxing, scanning and copying processes on synthetic paper.

# Overview

Training neural networks to work with images requires us to augment them in a variety of ways so that they learn to generalize. Networks designed to work with scanned document images must be trained with images that have the type of distortions seen in the wild.

Augraphy is an augmentation library developed to emulate these effects with a pipeline designed to simulate real world document processes.

# How It Works

Augraphy's augmentation pipeline starts with an image of clean text, like you might make in a document editor, and a list of effects like Ink Bleed, Dusty Ink and Low Ink. Augraphy then virtually prints that text, with the desired effects, to paper that has been generated mathematically or cropped from images of paper textures. This image is then augmented further with distortions that can be created by scanners, printers, or physical deformations.

The end result is an image that mimics real documents.

![Augmentation Pipeline](images/Pipeline.png)

**Example**

![Original Image](https://raw.githubusercontent.com/sparkfish/augraphy/dev/images/AugraphyExampleInput.png)
![Augmented Image](https://raw.githubusercontent.com/sparkfish/augraphy/dev/images/AugraphyExampleOutput.png)


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

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


# Citations
If you use Augraphy in your research, please cite the project:

```
@misc{Augraphy,
  author = {Sparkfish LLC},
  title = {Augmentation pipeline for rendering synthetic paper printing, faxing, scanning and copy machine processes.},
  howpublished = {\url{https://github.com/sparkfish/augraphy}},
}
```

# License
Copyright 2021 Sparkfish LLC

Augraphy is free and open-source software distributed under the terms of the [**MIT**](https://github.com/sparkfish/augraphy/blob/dev/LICENSE) license.

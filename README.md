# Augraphy

Augmentation pipeline for rendering synthetic paper printing and scanning processes.

Turn an image like this:

![Original Image](https://raw.githubusercontent.com/sparkfish/augraphy/dev/images/original.png)

Into an image like this:

![Augmented Image](https://raw.githubusercontent.com/sparkfish/augraphy/dev/images/augmented.png)

## Overview

Training neural networks to work with images requires us to augment them in a variety of ways so that they learn to generalize. Networks designed to work with scanned document images must be trained with images that have the type of distortions seen in the wild.

Augraphy is an augmentation library developed to emulate these effects in a pipeline designed to emulate the real world process of printing and scanning a document.

## How It Works

![Augmentation Pipeline](images/Augmentation.png)

Augraphy's augmentation pipeline starts with augmentations that emulate effects like Ink Bleed, Dusty Ink and Low Ink. Augraphy then virtually prints it to paper that has been generated mathematically or cropped from paper texture images. This image is then augmented further with distortions that can be created by scanners.

The end result is an image that mimics real scanned document images.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install augraphy.

```bash
pip install augraphy
```

## Usage

```python
from Augraphy import AugraphyPipeline

img = cv2.imread("test.png")
pipeline = AugraphyPipeline(paper_texture_path="./paper_textures")
crappified, original = pipeline.crappify(img, rotate=False)
```

```paper_texture_path``` defines where the images used for non-generated paper textures will be loaded from. See the ```paper_textures``` folder on Github for examples.

You can also enable debugging mode by setting the ```debug=True``` argument.

```pipeline = AugraphyPipeline(debug=True)```

This will use cv2.imshow for each image transformation or cv2_imshow if running in Google Colab.

## Roadmap

- [X] Ink effects (Bleed/Dusty/Low)
- [X] Paper textures (Generated/Images)
- [X] Scanner effects (Scanlines/Brightness Gradients)
- [ ] Improved Configuration capabilities

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
**MIT**

Copyright 2021 Sparkfish LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
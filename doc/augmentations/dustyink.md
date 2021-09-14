# Dusty Ink

The Dusty Ink augmentation applies random noise to the ink itself, emulating a dusty or inconsistent ink tone when followed by a blur.

**Example Usage:**

```python
augmentation = DustyInk(
	intensity_range=(.1, .2),
	color_range=(0, 224),
	p=0.5
    )
```

| Parameter         | Description                                                                                                                                |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `intensity_range` | Range of intensities to select from. Intensity must be a value between 0 to 1 and specifies the intensity of the noise added to the edges. |
| `color_range`     | The value range of the colors used for noise.                                                                                              |
| `p`               | The probability that this augmentation will be applied.                                                                                    |

**Example Result:**

Before and After Blur

![Dusty Ink no Blur](../../images/Augmentations/DustyInk.png)
![Dusty Ink with Blur](../../images/Augmentations/DustyInkBlur.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1vpuBElXkvhjzf3Z0Ou9ew1GwD7gRd5Ac?usp=sharing)

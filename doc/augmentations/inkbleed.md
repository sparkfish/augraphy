# Ink Bleed

The Ink Bleed augmentation relies on Sobel edge detection to create a mask of all edges, then applies random noise to those edges. When followed by a blur, this creates a fuzzy edge that emulates an ink bleed effect.

**Example Usage:**

```python
augmentation = InkBleed(
	intensity_range=(.1, .2),
	color_range=(0, 224)
	p=0.5
)
```

| Parameter | Description |
|---|---|
| `intensity_range` | Range of intensities to select from. Intensity must be a value between 0 to 1 and specifies the intensity of the noise added to the edges. |
| `color_range` | The value range of the colors used for noise. |
| `p` | The probability that this augmentation will be applied. |

**Example Result:**

Before and After Blur

![Ink Bleed no Blur](../../images/Augmentations/InkBleed.png)
![Ink Bleed with Blur](../../images/Augmentations/InkBleedBlur.png)


# Brightness Texturize

The Brightness Texturize augmentation creates a random noise in the brightness channel to emulate paper textures.

**Example Usage:**

```python
augmentation = BrightnessTexturize(
	range=(0.9, 0.99),
	deviation=0.03,
	p=0.5
    )
```

| Parameter   | Description                                             |
|-------------|---------------------------------------------------------|
| `range`     | The range of the brightness noise.                      |
| `deviation` | The deviation in the brightness noise.                  |
| `p`         | The probability that this augmentation will be applied. |

**Example Result:**

![Ink Bleed with Blur](../../images/Augmentations/BrightnessTexturizeAfter.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1vi4Tytxw-4Ap1pmSULtp3UhfjU8O9G2t?usp=sharing)

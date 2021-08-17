# Brightness Texturize

The Brightness Texturize augmentation creates a random noise in the brightness channel to emulate paper textures.

**Example Usage:**

```python
augmentation = BrightnessTexturize(
	range=(0.9, 0.99),
	deviation=0.03
	p=0.5
    )
```

| Parameter | Description |
|---|---|
| `range` | The range of the brightness noise. |
| `deviation` | The deviation in the brightness noise. |
| `p` | The probability that this augmentation will be applied. |

**Example Result:**

![Ink Bleed with Blur](images/Augmentations/BrightnessTexturizeAfter.png)

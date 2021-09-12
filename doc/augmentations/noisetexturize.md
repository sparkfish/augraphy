# Noise Texturize

The Noise Texturize augmentation creates a random noise based texture pattern to emulate paper textures.

**Example Usage:**

```python
augmentation = NoiseTexturize(
	sigma_range=(3, 10),
	turbulence_range=(2, 5),
	p=0.5
    )
```

| Parameter          | Description                                                                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sigma_range`      | The bounds of noise fluctuations.                                                                                                                    |
| `turbulence_range` | How quickly big patterns will be replaced with the small ones. The lower the value, the more iterations will be performed during texture generation. |
| `p`                | The probability that this augmentation will be applied.                                                                                              |

**Example Result:**

![Ink Bleed with Blur](../../images/Augmentations/NoiseTexturizeAfter.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1UOwbY83BeEbpRXiSQlhZvodUvrUVPpXh?usp=sharing)

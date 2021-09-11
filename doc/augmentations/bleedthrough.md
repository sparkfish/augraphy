# BleedThrough

The BleedThrough augmentation emulates ink bleeding through the page from the reverse side.

**Example Usage:**

```python
augmentation = BleedThrough(
	intensity_range=(0.1, 0.2),
	color_range=(0,224),
	ksize=(17,17),
	p=1
	)
```

| Parameter         | Description                                                                                                           |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|
| `intensity_range` | The range from which noise intensity is sampled.                                                                      |
| `color_range`     | The range from which color noise is sampled.                                                                          |
| `ksize`           | Tuple of height/width from which to sample the kernel size. Higher value increases the spread of the bleeding effect. |
| `sigmaX`          | Standard deviation of the kernel along the x-axis.                                                                    |
| `alpha`           | Intensity of bleeding effect.                                                                                         |
| `offsets`         | Distance in x and y directions to shift the bleedthrough effect.                                                      |
| `p`               | The probability this augmentation will be applied.                                                                    |

# Brightness

The Brightness augmentation adjusts the brightness of the whole image by a chosen multiplier.

**Example Usage:**

```python
augmentation = Brightness(
	range=(0.8, 1.4),
	p=0.5
    )
```

| Parameter | Description                                                                       |
|-----------|-----------------------------------------------------------------------------------|
| `range`   | The range of values to be chosen at random for the brightness multiplier applied. |
| `p`       | The probability that this augmentation will be applied.                           |

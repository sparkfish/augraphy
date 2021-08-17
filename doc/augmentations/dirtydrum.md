# Dirty Drum

The Dirty Drum augmentation emulates deposits of dirt and ink-grime from dirty printer drums.

**Example Usage:**

```python
augmentation = DirtyDrum(
	line_width_range=(6,18),
	ksize=(17,17),
	p=1
	)
```

| Parameter | Description |
| `line_width_range` | The range from which the width of a dirty drum line is sampled. |
| `ksize` | Height/width pair from which to sample the kernel size. |
| `sigmaX` | Standard deviation of the kernel along the x-axis. |
| `alpha` | Intensity of the effect. |
| `p` | The probability this augmentation will be applied. |

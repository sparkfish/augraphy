# Dirty Rollers

The Dirty Rollers augmentation emulates an effect created by certain document scanners.

**Example Usage:**

```python
augmentation = DirtyRollers(
	line_width_range=(8, 12),
	p=0.5
    )
```

| Parameter          | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| `line_width_range` | The base width of the rollers/bars/lines of the brightness gradients. |
| `p`                | The probability that this augmentation will be applied.               |

**Example Result:**

![Ink Bleed no Blur](../../images/Augmentations/DirtyRollers.png)

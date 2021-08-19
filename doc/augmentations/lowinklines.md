# Low Ink Lines

`LowInkRandomLines` and `LowInkPeriodicLines` inherit from `LowInkLine`. LowInkRandomLines adds low ink lines randomly throughout the image while LowInkPeriodicLines creates a set of lines that repeat in a periodic fashion throughout the image.

**Example Usage:**

```python
augmentation = LowInkRandomLines(
	count_range=(5, 10),
	use_consistent_lines=True,
	p=0.5
    )
```

| Parameter | Description |
|---|---|
| `count_range` | The number of lines to add to the image. |
| `p` | The probability that this augmentation will be applied. |

```python
augmentation = LowInkPeriodicLines(
	count_range=(5, 10),
	period_range=(10, 30),
	use_consistent_lines=True,
	p=0.5
    )
```

| Parameter | Description |
|---|---|
| `count_range` | The number of lines to add that will be repeated. |
| `period_range` | The number of pixels in each period before lines are repeated.|
| `p` | The probability that this augmentation will be applied. |

**Example:**

![Ink Bleed no Blur](../../images/Augmentations/LowInkLines.png)
![Ink Bleed with Blur](../../images/Augmentations/LowInkLinesBlur.png)

# BookBinding

The BookBinding augmentation creates the effect of a page curling away from the scanner bed, towards the binding of a book or magazine.

**Example Usage:**

```python
augmentation = BookBinding(
	radius_range=(1, 100),
	curve_intensity_range=(0,70),
	p=0.5
	)
```

| Parameter               | Description                                             |
|-------------------------|---------------------------------------------------------|
| `radius_range`          | The range of the curl radius, in pixels.                |
| `curve_intensity_range` | The intensity by which the page text should be curved.  |
| `p`                     | The probability that this augmentation will be applied. |

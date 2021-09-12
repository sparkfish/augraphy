# Strikethrough

The Strikethrough augmentation adds a smooth drawn line through a region of text characters.

**Example Usage:**

```python
augmentation = Strikethrough(
	num_lines_range=(2, 7),
	strikethrough_length_range=(0.5, 1),
	strikethrough_thickness_range=(1, 3),
	p=0.5,
	)
```

| Parameter                       | Description                                                                         |
|---------------------------------|-------------------------------------------------------------------------------------|
| `num_lines_range`               | Pair of ints determining the number of lines to add the effect to.                  |
| `strikethrough_length_range`    | Pair of floats between 0 and 1, determining the length of the strikethrough effect. |
| `strikethrough_thickness_range` | Pair of ints, to determine the thickness of strikethrough line.                     |
| `p`                             | The probability this augmentation will be applied.                                  |

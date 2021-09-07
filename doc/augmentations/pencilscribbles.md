# Pencil Scribbles

The Pencil Scribbles augmentation applies random pencil scribbles to image.

**Example Usage:**

```python
augmentation = PencilScribbles(
    size_range=(250, 400),
    count_range=(1, 10),
    stroke_count_range=(3, 6),
    thickness_range=(2, 6),
    brightness_change=128,
    p=1.0)
```

| Parameter | Description |
|---|---|
| `size_range` | The range for the size of the scribble to be created |
| `count_range` | The range for the number of scribbles to create. |
| `stroke_count_range` | The range for the number of strokes to create in each scribble. |
| `thickness_range` | The range for the size of the scribbles to create. |
| `brightness_change` | Value change for the brightness of the strokes. Default 128 creates a graphite-like appearance. |
| `p` | The probability that this augmentation will be applied. |

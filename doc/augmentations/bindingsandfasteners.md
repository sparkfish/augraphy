# Bindings And Fasteners

The Bindings And Fasteners augmentation creates binding and fastener mark in the input image.

**Example Usage:**

```python
augmentation = BindingsAndFasteners(
    overlay_types = "mix",
	foreground=None,
	effect_type = "punch_holes",
	ntimes = 3,
	edge = "left",
	edgeOffset= 50,
	p=0.5,
	)
```

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `overlay_types` | Types of overlay method, min, max or mix.                        |
| `foreground`    | Path to foreground image.                                        |
| `effect_type`   | Types of binding effect.                                         |
| `ntimes`        | Number of repetition to draw foreground image.                   |
| `edge`          | Which edge of the page the foreground copies should be placed on.|
| `edge_offset`   | How far from the edge of the page to draw the copies.            |
| `p`             | The probability this augmentation will be applied.               |

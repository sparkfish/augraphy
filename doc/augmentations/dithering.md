# Dithering

The Dithering augmentation applies Ordered or Floyd Steinberg dithering to the input image.

**Example Usage:**

```python
augmentation = Dithering(
	layer="ink",
	dither="ordered",
	order=5,
	p=0.5,
    )
```

| Parameter | Description                                               |
|-----------|-----------------------------------------------------------|
| `layer`   | The layer of image to apply dithering.                    |
| `dither`  | Types of dithering, ordered or Floyd Steinberg dithering. |
| `order`   | The order number for ordered dithering.                   |
| `p`       | The probability that this augmentation will be applied.   |

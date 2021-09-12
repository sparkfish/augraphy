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

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/19bqVyTF21BlzvEH4wOg_zFoFnCepPH1h?usp=sharing)

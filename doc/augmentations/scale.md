# Scale

Scales the image up or down.

**Example Usage:**

```python
augmentation = Scale(
	layer="post",
	scale_factor=1.5,
	p=1
	)
```

| Parameter       | Description                                             |
|-----------------|---------------------------------------------------------|
| `layer`         | The pipeline layer this image will be applied in.       |
| `scale_factor`  | How much to scale the image by.                         |
| `interpolation` | The interpolation algorithm to use when scaling.        |
| `p`             | The probability that this augmentation will be applied. |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1M7GnELBb7SNN1tzzXXD3U4WiMZOTHBgW?usp=sharing)

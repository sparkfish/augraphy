# Gaussian Blur

The Gaussian Blur augmentation applies a Gaussian blur to the whole image.

**Example Usage:**

```python
augmentation = GaussianBlur(
	kernels=[(3,3)],
	sigmaX=0,
	p=0.5
    )
```

| Parameter | Description                                                                                        |
|-----------|----------------------------------------------------------------------------------------------------|
| `kernels` | Specifies a list of blur kernels, one of which will be selected randomly when the blur is applied. |
| `sigmaX`  | Standard deviation of the kernel along the x-axis.                                                 |
| `p`       | The probability that this augmentation will be applied.                                            |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1bmbgDjxjwWUO-LKTPaZNL2xywam-NgWJ?usp=sharing)

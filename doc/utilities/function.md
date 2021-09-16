# Function

This augmentation wraps an arbitrary function or list of arbitrary functions.

**Example Usage:**

```python
def scaleUp2(img):
	return cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

augmentation = Function(
	scaleUp2,
	layer="post",
	p=0.5,
	)
```

| Parameter | Description                                             |
|-----------|---------------------------------------------------------|
| `fs`      | The function or list of functions to apply.             |
| `layer`   | The pipeline layer this augmentation is applied in.     |
| `p`       | The probability that this augmentation will be applied. |

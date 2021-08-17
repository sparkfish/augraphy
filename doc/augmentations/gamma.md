# Gamma Adjustment

The Gamma augmentation randomly applies gamma correction (from a range of values) on the image.
Note: a gamma value of 1 will not affect the image.

**Example Usage:**

```python
augmentation = Gamma(
	range=(0.5,1.5),
	p=0.5
    )
```

| Parameter | Description |
|---|---|
| `range` | Specifies the range for gamma value. Gamma value below 1 darkens the image whereas above 1 increases pixel intensities. Gamma value of 1 does not change anything. |
| `p` | The probability that this augmentation will be applied. |

**Example Result:**

In the image below, the leftmost picture is the original, the middle image is obtained by using a gamma value of 1.4, and the rightmost image is obtained by using a gamma value of 0.6.

![Gamma](images/Augmentations/gamma.png)

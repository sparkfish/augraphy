# Letterpress

The Letterpress augmentation produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

**Example Usage:**

```python
augmentation = Letterpress(
	count_range=(1000, 2500),
	size_range=(60, 80),
	points_range=(200, 250),
	p=1
	)
```

| Parameter        | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| `count_range`    | The range from which the number of blobs to apply is sampled.                    |
| `size_range`     | The range from which the blob diameter is sampled.                               |
| `points_range`   | The range from which the number of points in a blob is sampled.                  |
| `std_range`      | The range from which the standard deviation of the blob distribution is sampled. |
| `features_range` | The range from which the number of features in the blob is sampled.              |
| `value_range`    | The range from which the value of a point in the blob is sampled.                |
| `p`              | The probability this augmentation will be applied.                               |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1qp_ZnbdP0A4eIhxkF0h7Dduw_9gtAhti?usp=sharing)

# Letterpress

The Letterpress augmentation produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

**Example Usage:**

```python
augmentation = Letterpress(
	n_samples = (200,300),
	n_clusters = (500,1000),
	std_range=(1000, 2000),
	value_range=(230, 255)
	p=1
	)
```

| Parameter        | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| `n_samples`      | Pair of ints determining number of points in a cluster.                          |
| `n_clusters`     | Pair of ints determining number of clusters.                                     |
| `std_range`      | The range from which the standard deviation of the blob distribution is sampled. |
| `value_range`    | The range from which the value of a point in the blob is sampled.                |
| `p`              | The probability this augmentation will be applied.                               |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1qp_ZnbdP0A4eIhxkF0h7Dduw_9gtAhti?usp=sharing)

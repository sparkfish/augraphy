# Folding

The Folding augmentation emulates folded paper being scanned, with a visible warp effect around the fold line.

**Example Usage:**

```python
augmentation = Folding(
	fold_count=3,
	fold_noise=0.15,
	p=0.5
	)
```

| Parameter         | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| `fold_count`      | Number of folds to apply.                                                                     |
| `fold_noise`      | Level of noise added to folding area. Range from 0 to 1.                                      |
| `gradient_width`  | Measure of the space affected by the fold prior to being warped, as percentage of page width. |
| `gradient_height` | Measure of depth of fold, as percentage of page height.                                       |
| `p`               | The probability that this augmentation will be applied.                                       |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1DNZoPc_z2CG8-VpwTvBWQoJG0cWa86MV?usp=sharing)

# Low Ink Blobs

The Low Ink Blobs augmentation creates random blobs of "low ink" that will be applied to the image.

**Example Usage:**

```python
augmentation = LowInkBlobs(
	count_range=(5, 25),
	size_range=(10, 20),
	points_range=(5, 25),
	std_range=(10, 75),
	features_range=(15, 25),
	value_range=(180, 250),
	p=0.5
	)
```

| Parameter        | Description                                                                        |
|------------------|------------------------------------------------------------------------------------|
| `count_range`    | The range for the number of blobs to add to the image.                             |
| `size_range`     | The range in pixels for the size of the image patch that blobs will be created in. |
| `points_range`   | The number of points to add to image patch to create the blob.                     |
| `std_range`      | The std_range value passed into `sklearn.datasets.make_blobs`                      |
| `features_range` | The features_range value passed into `sklearn.datasets.make_blobs`                 |
| `values_range`   | The range of values used for the blob pixels.                                      |
| `p`              | The probability that this augmentation will be applied.                            |

**Example Result:**

![Ink Bleed no Blur](../../images/Augmentations/LowInkBlobs.png)
![Ink Bleed with Blur](../../images/Augmentations/LowInkBlobsBlur.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/18htEZ2nkdZsq1vJgZaIEogzODv820sRy?usp=sharing)

# Page Border

The Page Border augmentation applies a shadow of underlying pages on any side of the page, creating an effect of single or multiple borders on specified side of the page.

**Example Usage:**

```python
augmentation = PageBorder(
	side="left",
	width_range=(10,30),
	pages=4,
	noise_intensity_range=(0.2,0.5),
	p=0.5
    )
```

| Parameter               | Description                                                                               |
|-------------------------|-------------------------------------------------------------------------------------------|
| `side`                  | The side of the page on which the border will be drawn. One of: left, right, top, bottom. |
| `width_range`           | The width of the border.                                                                  |
| `pages`                 | The number of pages to augment in the border.                                             |
| `noise_intensity_range` | The range of noise applied around the border and pages.                                   |
| `p`                     | The probability that this augmentation will be applied.                                   |

**Example Result:**

![Border](../../images/Augmentations/PageBorder.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1eRFCmmyoQD87ZgAeweAk5dkygS-Hk3Zu?usp=sharing)

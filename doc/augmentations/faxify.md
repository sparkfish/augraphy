# Faxify

The faxify augmentation emulates faxify effect in the image.

**Example Usage:**

```python
augmentation = Faxify(
			     scale_range=(1, 1), 
                 monochrome=0,
                 monochrome_method="Otsu",
                 monochrome_threshold=127,
                 invert=1,
                 half_kernel_size=2,
                 angle=45,
                 sigma=2,
                 p=0.5,
    )
```

| Parameter | Description |
|---|---|
| `scale_range` | Pair of ints determining the range from which to divide the resolution by. |
| `monochrome` | Flag to apply monochrome effect, false to select halftone effect. |
| `monochrome_method` | Otsu, Simple or Adaptive method. |
| `monochrome_threshold` | The simple binarization threshold value. |
| `invert` | Invert grayscale value in halftone effect. |
| `half_kernel_size` | The half size of gaussian kernel for halftone effect. |
| `angle` | The angle of halftone effect. |
| `sigma` | The sigma value of gaussian kernel in halftone effect. |
| `p` | The probability that this augmentation will be applied. |

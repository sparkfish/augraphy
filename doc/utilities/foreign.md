# Foreign

The Foreign class provides a means of wrapping augmentations from other projects.

Currently, [Albumentations](https://github.com/albumentations-team/albumentations) and [imgaug](https://github.com/aleju/imgaug) are supported.

```python
from albumentations.augmentations.transforms import VerticalFlip

augmentation = Foreign(
	foreignAugmentation=VerticalFlip(),
	p=1
	)
```

| Parameter             | Description                                             |
|-----------------------|---------------------------------------------------------|
| `foreignAugmentation` | The Albumentations or imgaug transform to apply.        |
| `p`                   | The probability that this augmentation will be applied. |

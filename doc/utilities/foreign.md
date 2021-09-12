# Foreign

The Foreign class provides a means of wrapping augmentations from other projects.

Currently, [Albumentations](https://github.com/albumentations-team/albumentations) and [imgaug](https://github.com/aleju/imgaug).

```python
from albumentations.augmentations.transforms import VerticalFlip

augmentation = Foreign(
	layer="post",
	foreignAugmentation=VerticalFlip(),
	p=1
	)
```

| Parameter             | Description                                      |
|-----------------------|--------------------------------------------------|
| `layer`               | The layer this augmentation is applied in.       |
| `foreignAugmentation` | The Albumentations or imgaug transform to apply. |

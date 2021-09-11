# Augmentation Sequence

Augmentation Sequence creates a sequence of augmentations that will be executed in order.

**Example Usage:**

```python
sequence = AugmentationSequence(
	augmentations=[
		# Add Augmentations Here
		Bleedthrough(p=1),
		GaussianBlur(p=0.75)
		],
	p=0.5
	)
```

| Parameter       | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `augmentations` | specifies the list of augmentations to be chosen from.             |
| `p`             | specifies the probability that the augmentation sequence will run. |

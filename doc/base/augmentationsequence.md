# Augmentation Sequence

Augmentation Sequence creates a sequence of augmentations that will be executed in order.

Sequences can be used as part of an Augraphy pipeline, within which we call them "phases", or can be used as standalone callables.

**Example Usage:**

```python
ink_phase = AugmentationSequence(
	augmentations=[
		# Add Augmentations Here
		Bleedthrough(),
		Brightness()
		],
	p=0.5
	)

pipeline = AugraphyPipeline(ink_phase, ...)
```


```python
sequence2 = AugmentationSequence([Bleedthrough(), Brightness()])

augmented = sequence2(image)
```

| Parameter       | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `augmentations` | specifies the list of augmentations to be chosen from.             |
| `p`             | specifies the probability that the augmentation sequence will run. |

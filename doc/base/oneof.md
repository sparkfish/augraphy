# OneOf

`OneOf` is a wrapper around a list of augmentations, which randomly picks one to execute.

`OneOf` may be used just like any other augmentation, either within a pipeline or as a standalone callable.

**Example Usage:**

```python
one_of = OneOf(
	augmentations=[
		# Add Augmentations Here
		Bleedthrough(),
		Brightness()
        ],
	p=0.5
    )

ink_phase = [one_of, Letterpress(p=0.5), Gamma(p=0.25)]

pipeline = AugraphyPipeline(ink_phase, ...)
```

```python
pick_one = OneOf([Bleedthrough(), Brightness(), Letterpress()])

augmented = pick_one(image)
```

| Parameter       | Description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| `augmentations` | specifies the list of augmentations to be chosen from.                                       |
| `p`             | specifies the probability that the augmentation will run one of the specified augmentations. |

# OneOf

Augmentation list that will execute one of the specified augmentations randomly. Probabilities for the specified augmentations will be used as weights for which one will be selected, and the probability at the OneOf level will be used to determine if any are selected.

**Example Usage:**

```python
augmentation = OneOf(
        augmentations=[
            # Add Augmentations Here
			Bleedthrough(p=1),
			GaussianBlur(p=0.75)
        ],
        p=0.5
    )
```

| Parameter  | Description  |
|---|---|
|   ```augmentations``` | specifies the list of augmentations to be chosen from.   |
|   ```p``` | specifies the probability that the augmentation will run one of the specified augmentations.   |


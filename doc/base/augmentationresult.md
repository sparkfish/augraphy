# Augmentation Result

Contains the result of an augmentation's application, and the augmentation itself. Augmentations add these to the pipeline data dictionary during pipeline execution.

** Example Usage: **
```python
data[layer].append(AugmentationResult(self, augmentation))
```
| Parameter | Description |
|---|---|
| `augmentation` | The augmentation that was applied. |
| `result` | The image transformed by `augmentation`. |
| `metadata` | Additional data that may be added by callers. |

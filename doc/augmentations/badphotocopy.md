# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

**Example Usage:**

```python
augmentation = BadPhotocopy(
	blob_density = 0.6,
	p=0.5
	)
```

| Parameter | Description |
|---|---|
| `blob_density` | Amount of blobs in the image, a range from 0 to 1. |
| `p` | The probability this augmentation will be applied. |

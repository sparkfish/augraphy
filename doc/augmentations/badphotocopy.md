# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

**Example Usage:**

```python
augmentation = BadPhotocopy(
	noise_density=(0.1, 0.9),
	max_iteration=(7, 9),
	hash_type=0,
	p=0.5
	)
```

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `noise_density` | Density of noise blobs, lower value generates sparser noise.     |
| `max_iteration` | Iterations of noise value, higher value generates sparser noise. |
| `hash_type`     | Types of hashes to generate olsen noise.                         |
| `p`             | The probability this augmentation will be applied.               |

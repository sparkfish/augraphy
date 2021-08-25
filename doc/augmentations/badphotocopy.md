# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

**Example Usage:**

```python
augmentation = BadPhotocopy(
	noise_density=(0.01, 0.1), 
	max_iteration=(12, 15),  
	p=0.5
	)
```

| Parameter | Description |
|---|---|
| `noise_density` | Density of noise blobs, lower value generates sparser noise. |
| `max_iteration` | Iterations of noise value, higher value generates sparser noise. |
| `p` | The probability this augmentation will be applied. |

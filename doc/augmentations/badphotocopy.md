# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

**Example Usage:**

```python
augmentation = BadPhotocopy(
	layer="ink",
	noise_density=(0.1, 0.9),
	max_iteration=(7, 9),
	hash_type=0,
	wave_pattern=1,
	p=0.5
	)
```

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `layer`		  | The image layer to apply the augmentation to.                    |
| `noise_density` | Density of noise blobs, lower value generates sparser noise.     |
| `max_iteration` | Iterations of noise value, higher value generates sparser noise. |
| `hash_type`     | Types of hashes to generate olsen noise.                         |
| `wave_pattern`  | To enable wave pattern in noise.                                 |
| `p`             | The probability this augmentation will be applied.               |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1_n7atcI-xVSpnV17YuzZhd9ISe735fBo?usp=sharing)

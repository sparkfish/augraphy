# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

**Example Usage:**

```python
augmentation = BadPhotocopy(
	nperiod = (4,4),
	octaves = 4,
	persistence = 0.5,
	lacunarity = 2,
	wave_pattern=1,
	p=0.5
	)
```

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `nperiod`		  | The number of periods of noise to generate along each axis.      |
| `octaves`       | The number of octaves in the noise.                              |
| `persistence`   | The scaling factor between two octaves.                          |
| `lacunarity`    | The frequency factor between two octaves.                        |
| `wave_pattern`  | To enable wave pattern in noise.                                 |
| `p`             | The probability this augmentation will be applied.               |

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1_n7atcI-xVSpnV17YuzZhd9ISe735fBo?usp=sharing)

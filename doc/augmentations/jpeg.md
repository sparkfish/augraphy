# JPEG Compression

Uses JPEG encoding to create JPEG compression artifacts in the image.

**Example Usage:**

```python
augmentation = Jpeg(
	quality_range=(50, 95),
	p=0.5
    )
```

| Parameter | Description |
|---|---|
| `quality_range` | The quality range for the JPEG compression encoding. |
| `p` | The probability that this augmentation will be applied. |

**Example Result:**

Encoded with quality range of (10, 15)

![JPEG Compression Before](images/Augmentations/JPEGCompressionBefore.png)
![JPEG Compression After](images/Augmentations/JPEGCompression.png)

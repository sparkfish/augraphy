# ImageOverlay

The ImageOverlay augmentation places a foreground image randomly over the document.

**Example Usage:**

```python
paperclip = cv2.imread("paperclip.png")

augmentation = ImageOverlay(
	foreground=paperclip,
	layer="post",
	p=0.5,
	)
```

| Parameter | Description |
| `foreground` | The image you want to overlay on the document. |
| `layer` | The layer of image to apply the overlay to. |
| `p` | The probability that this augmentation will be applied. |

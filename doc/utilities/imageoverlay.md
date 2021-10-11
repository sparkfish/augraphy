# ImageOverlay

The `ImageOverlay` augmentation places a foreground image on the document, at a specified location or randomly.

**Example Usage:**

```python
paperclip = cv2.imread("paperclip.png")

augmentation = ImageOverlay(
	foreground=paperclip,
	position=(350,170)
	p=0.5,
	)
```

| Parameter    | Description                                             |
|--------------|---------------------------------------------------------|
| `foreground` | The image you want to overlay on the document.          |
| `position`   | Pair of coordinates to place the image at, optionally   |
| `p`          | The probability that this augmentation will be applied. |

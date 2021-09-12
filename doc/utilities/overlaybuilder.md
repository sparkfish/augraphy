# OverlayBuilder

The OverlayBuilder class creates an image mask containing other images, like hole punches or binder clips, to be overlaid on a document.

**Example Usage:**

```python
bg = np.ones((1500,1000,3))
fg = np.zeros((50,150,3))

ob = OverlayBuilder(fg, bg, 5, "left", 10)
overlay = ob.buildOverlay()
```

| Parameter    | Description                                                      |
|--------------|------------------------------------------------------------------|
| `foreground` | the image to overlay on the background document                  |
| `background` | the document to overlay on                                       |
| `ntimes`     | how many copies of the foreground image to draw                  |
| `edge`       | which edge of the page the foreground copies should be placed on |
| `edgeOffset` | how far from the edge of the page to draw the copies             |

# Paper Factory

`PaperFactory` randomly replaces the starting paper image with a texture chosen from a directory, resized or cropped and tiled to fit the dimensions of the input image.

`PaperFactory` can be used as part of an Augraphy pipeline (within the paper phase), or can be used as a standalone callable. In the latter case, it returns the paper image.

**Example Usage:**

```python
pf1 = PaperFactory(
	tile_texture_shape=(250,250),
	texture_path="./paper_textures"
	p=0.5
    )

paper_phase = [pf1, BrightnessTexturize(p=0.5)]
```

```python
pf2 = PaperFactory()

paper_image = pf2(input_image)
```

| Parameter            | Description                                                                                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `tile_texture_shape` | specifies the size of the texture crop when tiling a texture.                                                                               |
| `paper_texture_path` | defines where the images used for non-generated paper textures will be loaded from. See the `paper_textures` folder on Github for examples. |
| `p`                  | specifies the probability that the augmentation will run.                                                                                   |

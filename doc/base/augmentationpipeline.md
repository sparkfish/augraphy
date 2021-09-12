# Augraphy Pipeline

The Augraphy Pipeline contains "phases" or "layers" of image augmentations and their results.

**Note:** Augraphy Pipelines only accept images **larger than 30x30 pixels.**

**Example Usage:**
```python
ink_phase = [InkBleed(), Letterpress(), GaussianBlur("ink")]

paper_phase = [PaperFactory(), NoiseTexturize(), BrightnessTexturize()]

post_phase = [DirtyRollers(), DirtyDrum, Jpeg()]

pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
```

| Parameter           | Description                                                             |
|---------------------|-------------------------------------------------------------------------|
| `ink_phase`         | Collection of augmentations to apply to the ink.                        |
| `paper_phase`       | Collection of augmentations to apply to the paper.                      |
| `post_phase`        | Collection of augmentations to apply to the printed paper.              |
| `ink_color_range`   | Pair of ints determining the range from which to sample ink color.      |
| `paper_color_range` | Pair of ints determining the range from which to sample paper color.    |
| `rotate_range`      | Pair of ints determining the range from which to sample paper rotation. |

## The Data Dictionary

The output of the pipeline will be a dictionary containing the image at various stages of processing along with augmentations applied. Additional metadata can be added by augmentations in the pipeline. In the example usage above, the `pipeline` variable contains this dictionary of results.

| Key                     | Description                                                                                                          |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|
| `data['image']`         | stores the initial input image before any modifications were made.                                                   |
| `data['image_rotated']` | stores the initial input image after rotation. This will serve as the *Ground Truth* for training neural networks.   |
| `data['ink']`           | contains a list of `AugmentationResult`s with the augmentation and resulting image for each step of the ink phase.   |
| `data['paper_texture']` | contains the image selected to be used as the paper texture by `PaperFactory`.                                       |
| `data['paper']`         | contains a list of `AugmentationResult`s with the augmentation and resulting image for each step of the paper phase. |
| `data['post']`          | contains a list of `AugmentationResult`s with the augmentation and resulting image for each step of the post phase.  |
| `data['output']`        | stores the final image after all augmentations are applied.                                                          |
| `data['log']`           | stores performance metadata and other information.                                                                   |

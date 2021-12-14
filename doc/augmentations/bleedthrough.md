# BleedThrough

The BleedThrough augmentation emulates ink bleeding through the page from the reverse side.


| Parameter         | Description                                                                                                           |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|
| `intensity_range` | The range from which noise intensity is sampled.                                                                      |
| `color_range`     | The range from which color noise is sampled.                                                                          |
| `ksize`           | Tuple of height/width from which to sample the kernel size. Higher value increases the spread of the bleeding effect. |
| `sigmaX`          | Standard deviation of the kernel along the x-axis.                                                                    |
| `alpha`           | Intensity of bleeding effect.                                                                                         |
| `offsets`         | Distance in x and y directions to shift the bleedthrough effect.                                                      |
| `p`               | The probability this augmentation will be applied.                                                                    |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500, 3), 255, dtype="uint8")

    # insert text into image
    for y in range(200, 1300, 100):
        cv2.putText(
            image,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            0,
            3,
        )

    bleedthrough = BleedThrough(
            intensity_range=(0.1, 0.2),
            color_range=(0, 224),
            ksize=(17, 17),
            sigmaX=0,
            alpha=0.2,
            offsets=(10, 20),
    )

    img_bleedthrough = bleedthrough(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(img_bleedthrough)
    plt.title("Blledthrough")

```

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1iwU2AnYhhJKEdddFP0Uu3DwEz8QnfQcs?usp=sharing)

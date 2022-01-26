# Dusty Ink

The Dusty Ink augmentation applies random noise to the ink itself, emulating a dusty or inconsistent ink tone when followed by a blur.



| Parameter         | Description                                                                                                                                |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `intensity_range` | Range of intensities to select from. Intensity must be a value between 0 to 1 and specifies the intensity of the noise added to the edges. |
| `color_range`     | The value range of the colors used for noise.                                                                                              |
| `value_range`     | Min value of pixel to enable dusty ink effect.                                                                                             |
| `p`               | The probability that this augmentation will be applied.                                                                                    |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500), 255, dtype="uint8")

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


    dusty_ink = DustyInk(
            intensity_range=(0.2, 0.4),
            color_range=(255, 255),
            value_range=(0,10)
            )

    img_dusty_ink = dusty_ink(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(img_dusty_ink, cmap='gray')
    plt.title("Dusty ink")
```


**Example Result:**

Before and After Blur

![Dusty Ink no Blur](../../images/Augmentations/DustyInk.png)
![Dusty Ink with Blur](../../images/Augmentations/DustyInkBlur.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vpuBElXkvhjzf3Z0Ou9ew1GwD7gRd5Ac?usp=sharing)

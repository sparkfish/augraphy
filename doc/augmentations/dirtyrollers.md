# Dirty Rollers

The Dirty Rollers augmentation emulates an effect created by certain document scanners.


| Parameter          | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| `line_width_range` | The base width of the rollers/bars/lines of the brightness gradients. |
| `scanline_type`    | Types of scanline, use 0 for white background.                        |
| `p`                | The probability that this augmentation will be applied.               |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500,3), 255, dtype="uint8")

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

    dirty_roller = DirtyRollers(
            line_width_range=(12, 25),
            scanline_type=0,
            )

    img_dirty_roller = dirty_roller(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_dirty_roller )
    plt.title("Dirty roller ")
```


**Example Result:**

![Ink Bleed no Blur](../../images/Augmentations/DirtyRollers.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G5z2_qRVIowcoRy3pQGiNLlWvQA0xwhS?usp=sharing)

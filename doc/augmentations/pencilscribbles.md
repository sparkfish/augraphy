# Pencil Scribbles

The Pencil Scribbles augmentation applies random pencil scribbles to image.


| Parameter            | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `size_range`         | The range for the size of the scribble to be created                                            |
| `count_range`        | The range for the number of scribbles to create.                                                |
| `stroke_count_range` | The range for the number of strokes to create in each scribble.                                 |
| `thickness_range`    | The range for the size of the scribbles to create.                                              |
| `brightness_change`  | Value change for the brightness of the strokes. Default 128 creates a graphite-like appearance. |
| `p`                  | The probability that this augmentation will be applied.                                         |


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


    pencil_scribbles = PencilScribbles(
        size_range=(400, 800),
        count_range=(2, 3),
        stroke_count_range=(1, 2),
        thickness_range=(1, 3),
        brightness_change=128
            )

    img_pencil_scribbles = pencil_scribbles(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_pencil_scribbles)
    plt.title("Image pencil scribbles")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jyBODoE2UxLAUVgkFEQ2s1gOScjCzpdk?usp=sharing)

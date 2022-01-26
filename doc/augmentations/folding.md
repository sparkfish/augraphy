# Folding

The Folding augmentation emulates folded paper being scanned, with a visible warp effect around the fold line.


| Parameter         | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| `fold_x`          | X coordinate of the folding effect.                                                           |
| `fold_deviation`  | Deviation (in pixels) of provided X coordinate location.                                      |
| `fold_count`      | Number of folds to apply.                                                                     |
| `fold_noise`      | Level of noise added to folding area. Range from 0 to 1.                                      |
| `gradient_width`  | Measure of the space affected by the fold prior to being warped, as percentage of page width. |
| `gradient_height` | Measure of depth of fold, as percentage of page height.                                       |
| `p`               | The probability that this augmentation will be applied.                                       |


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


    folding = Folding(
        fold_count=2,
        fold_noise=0.0,
        gradient_width=(0.2, 0.3),
        gradient_height=(0.01, 0.02),
            )

    img_folded= folding(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_folded)
    plt.title("Folded Image")
```


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DNZoPc_z2CG8-VpwTvBWQoJG0cWa86MV?usp=sharing)

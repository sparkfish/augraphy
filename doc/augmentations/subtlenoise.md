# Subtle Noise

The Subtle Noise augmentation emulates the imperfections in scanning solid colors due to subtle lighting differences.


| Parameter | Description                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `range`   | The range added or subtracted from each pixel value in the image. With a range of 5, a pixel with color value of 100 will end up between 95 and 105. |
| `p`       | The probability that this augmentation will be applied.                                                                                              |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

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


    subtle_noise = SubtleNoise(
        range=25,
            )

    img_subtle_noise = subtle_noise(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_subtle_noise)
    plt.title("Image subtle_noise")
```


**Example Result:**

Created with a range of 25.

![Subtle Noise](../../images/Augmentations/SubtleNoiseExample.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1mod73_cQ5qlmJm3ppHLimpQntQUmMPIZ?usp=sharing)

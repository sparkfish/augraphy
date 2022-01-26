# JPEG Compression

Uses JPEG encoding to create JPEG compression artifacts in the image.


| Parameter       | Description                                             |
|-----------------|---------------------------------------------------------|
| `quality_range` | The quality range for the JPEG compression encoding.    |
| `p`             | The probability that this augmentation will be applied. |


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


    jpeg_5 = Jpeg(
            quality_range=(5, 5)
            )

    jpeg_95 = Jpeg(
            quality_range=(95, 95)
            )

    img_jpeg_5 = jpeg_5(image)
    img_jpeg_95 = jpeg_95(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_jpeg_5)
    plt.title("Jpeg 5% quality")
    plt.subplot(122)
    plt.imshow(img_jpeg_95)
    plt.title("Jpeg 95% quality")
```


**Example Result:**

Encoded with quality range of (10, 15)

![JPEG Compression Before](../../images/Augmentations/JPEGCompressionBefore.png)
![JPEG Compression After](../../images/Augmentations/JPEGCompression.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mlv1jb8PTes9bKz5i0C3b353WK8kgWeH?usp=sharing)

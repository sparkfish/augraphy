# Low Ink Blobs

The Low Ink Blobs augmentation creates random blobs of "low ink" that will be applied to the image.


| Parameter        | Description                                                                        |
|------------------|------------------------------------------------------------------------------------|
| `count_range`    | The range for the number of blobs to add to the image.                             |
| `size_range`     | The range in pixels for the size of the image patch that blobs will be created in. |
| `points_range`   | The number of points to add to image patch to create the blob.                     |
| `std_range`      | The std_range value passed into `sklearn.datasets.make_blobs`                      |
| `features_range` | The features_range value passed into `sklearn.datasets.make_blobs`                 |
| `values_range`   | The range of values used for the blob pixels.                                      |
| `p`              | The probability that this augmentation will be applied.                            |


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


    low_ink_blobs = LowInkBlobs(
        count_range=(400, 500),
        size_range=(50, 100),
        points_range=(100, 150),
        std_range=(10, 75),
        features_range=(15, 25),
        value_range=(180, 250)
            )

    img_low_ink_blobs = low_ink_blobs(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_low_ink_blobs)
    plt.title("Image low_ink_blobs")
```


**Example Result:**

![Ink Bleed no Blur](../../images/Augmentations/LowInkBlobs.png)
![Ink Bleed with Blur](../../images/Augmentations/LowInkBlobsBlur.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/18htEZ2nkdZsq1vJgZaIEogzODv820sRy?usp=sharing)

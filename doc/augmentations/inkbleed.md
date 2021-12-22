# Ink Bleed

The Ink Bleed augmentation relies on Sobel edge detection to create a mask of all edges, then applies random noise to those edges. When followed by a blur, this creates a fuzzy edge that emulates an ink bleed effect.


| Parameter         | Description                                                                                                                                |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `intensity_range` | Range of intensities to select from. Intensity must be a value between 0 to 1 and specifies the intensity of the noise added to the edges. |
| `color_range`     | The value range of the colors used for noise.                                                                                              |
| `kernel_size`     | Kernel size to determine area of inkbleed effect.                                                                                          |
| `severity`        | Severity to determine concentration of inkbleed effect.                                                                                    |
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
            "Lorem ipsum",
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            0,
            15,
        )

    inkbleed = InkBleed(
        intensity_range=(0.1, 0.2),
        color_range=(0, 224),
        kernel_size=(7, 7),
        severity=(0.3, 0.4)
    )

    img_inkbleed = inkbleed(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.subplot(121)
    plt.imshow(image,cmap='gray')
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(img_inkbleed,cmap='gray')
    plt.title("Inkbleed image")
```


**Example Result:**

Before and After Blur

![Ink Bleed no Blur](../../images/Augmentations/InkBleed.png)
![Ink Bleed with Blur](../../images/Augmentations/InkBleedBlur.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1hQr4eW2H0NNr1ov6Nw7sEHOf5QTJFOyX?usp=sharing)

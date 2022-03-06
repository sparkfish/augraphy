# Faxify

The Faxify augmentation emulates the artifacts created by faxing the document.


| Parameter              | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| `scale_range`          | Pair of ints determining the range from which to divide the resolution by. |
| `monochrome`           | Flag to enable monochrome effect.                                          |
| `monochrome_method`    | Monochrome thresholding method.                                            |
| `monochrome_arguments` | A dictionary contains argument to monochrome thresholding method.          |
| `halftone`             | Flag to enable halftone effect.                                            |
| `invert`               | Invert grayscale value in halftone effect.                                 |
| `half_kernel_size`     | The half size of gaussian kernel for halftone effect.                      |
| `angle`                | The angle of halftone effect.                                              |
| `sigma`                | The sigma value of gaussian kernel in halftone effect.                     |
| `p`                    | The probability that this augmentation will be applied.                    |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500,3), 0, dtype="uint8")

    # insert text into image
    for y in range(200, 1300, 100):
        cv2.putText(
            image,
            "Lorem ipsum dolor sit amet",
            (70, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255,255,255),
            15,
        )

    # create image with random colour text
    image = image * (np.random.random((1500,1500,3))*255).astype('uint8')

    arguments1 = {"thresh":128, "maxval":128, "type":cv2.THRESH_BINARY}
    faxify1 = Faxify(scale_range = (1,2),
                    monochrome = 1,
                    monochrome_method = "cv2.threshold",
                    monochrome_arguments = arguments1,
                    halftone = 1,
                    invert = 1,
                    half_kernel_size = 2,
                    angle = 45,
                    sigma = 1)

    arguments2 = {"window_size":99, "k":0.2}
    faxify2 = Faxify(scale_range = (1,2),
                    monochrome = 1,
                    monochrome_method = "threshold_niblack",
                    monochrome_arguments = arguments2,
                    halftone = 1,
                    invert = 1,
                    half_kernel_size = 2,
                    angle = 45,
                    sigma = 1)


    img_faxify1 = faxify1(image)
    img_faxify2 = faxify2(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(img_faxify1)
    plt.title("Faxify cv threshold")

    plt.figure()
    plt.imshow(img_faxify1)
    plt.title("Faxify skimage niblack")
```


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12hxA5JDCi76hxmjFTotw0A3JmnwNpL6i?usp=sharing)

# Faxify

The Faxify augmentation emulates the artifacts created by faxing the document.


| Parameter              | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| `scale_range`          | Pair of ints determining the range from which to divide the resolution by. |
| `monochrome`           | Flag to apply monochrome effect, false to select halftone effect.          |
| `monochrome_method`    | Otsu, Simple or Adaptive method.                                           |
| `adaptive_method`      | cv2 adaptive methods when adaptive method is used.                         |
| `monochrome_threshold` | The simple binarization threshold value.                                   |
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

    faxify_otsu = Faxify(
            scale_range=(4, 4),
            monochrome=1,
            monochrome_method="Otsu",
            )

    faxify_simple = Faxify(
            scale_range=(3, 3),
            monochrome=1,
            monochrome_method="Simple",
            monochrome_threshold=10,
            )

    faxify_adaptive = Faxify(
            scale_range=(2, 2),
            monochrome=1,
            monochrome_method="Adaptive",
            )

    faxify_halftone = Faxify(
            scale_range=(1, 1),
            monochrome=0,
            invert=0,
            half_kernel_size=4,
            angle=45,
            sigma=2,
            )

    img_faxify_otsu = faxify_otsu(image)
    img_faxify_simple = faxify_simple(image)
    img_faxify_adaptive = faxify_adaptive(image)
    img_faxify_halftone = faxify_halftone(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(img_faxify_otsu)
    plt.title("Faxify Otsu")

    plt.figure()
    plt.imshow(img_faxify_simple)
    plt.title("Faxify simple")

    plt.figure()
    plt.imshow(img_faxify_adaptive)
    plt.title("Faxify adaptive")

    plt.figure()
    plt.imshow(img_faxify_halftone)
    plt.title("Faxify halftone")
```


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12hxA5JDCi76hxmjFTotw0A3JmnwNpL6i?usp=sharing)

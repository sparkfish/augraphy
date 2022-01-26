# BookBinding

The BookBinding augmentation creates the effect of a page curling away from the scanner bed, towards the binding of a book or magazine.


| Parameter               | Description                                                |
|-------------------------|------------------------------------------------------------|
| `radius_range`          | The range of the curl radius, in pixels.                   |
| `curve_intensity_range` | The intensity by which the page text should be curved.     |
| `mirror`                | Flag to enable mirror effect.                              |
| `mirror_range`          | Tuple of floats to determine width of image to be mirrored.|
| `p`                     | The probability that this augmentation will be applied.    |


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

    book_binder = BookBinding(
            radius_range=(50, 150),
            curve_intensity_range=(30, 100)
            )

    img_book_binding= book_binder(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_book_binding)
    plt.title("Book Binding")

```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ju1kzxpHhIZWUbgML8PKaw8KMBoz3PTC?usp=sharing)

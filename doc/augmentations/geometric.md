# Geometric

The geometric augmentation applies basic geometric transformations such as resizing, flips and rotation.


| Parameter     | Description                                                                       |
|---------------|-----------------------------------------------------------------------------------|
| `scale`       | Pair of floats determining new scale of image.                                    |
| `fliplr`      | Flag to flip image in left right direction.                                       |
| `flipud`      | Flag to flip image in up down direction.                                          |
| `crop`        | Tuple of 4 (x0, y0, xn, yn) to crop section of image.                             |
| `rotate_range`| Pair of ints determining the range from which to sample the image rotation.       |
| `p`           | The probability that this augmentation will be applied.                           |


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


    geometric = Geometric(scale = (0.5,1.5),
                         fliplr = 1,
                         flipud = 1,
                         rotate_range = (-15,15))

    img_geometric_transform = geometric(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(img_geometric_transform)
    plt.title("Geometric transformed image")
```

# Dithering

The Dithering augmentation applies Ordered or Floyd Steinberg dithering to the input image.


| Parameter | Description                                               |
|-----------|-----------------------------------------------------------|
| `layer`   | The layer of image to apply dithering.                    |
| `dither`  | Types of dithering, ordered or Floyd Steinberg dithering. |
| `order`   | The order number for ordered dithering.                   |
| `p`       | The probability that this augmentation will be applied.   |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import cv2
    import numpy as np

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500,3), 255, dtype="uint8")

    # insert text into image
    for y in range(200, 1300, 100):
        cv2.putText(
            image,
            "Lorem ipsum",
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (128,128,128),
            20,
        )

    dirther_ordered = Dithering(
            dither="ordered",
            order=3,
            )

    dirther_floyd = Dithering(
            dither="floyd"
            )

    img_dither_ordered = dirther_ordered(image)
    img_dither_floyd  = dirther_floyd (image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(img_dither_ordered)
    plt.title("Ordered dithering")

    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(img_dither_floyd)
    plt.title("Floyd Steinberg dithering")
```


Click here for an [example Colab Notebook](https://colab.research.google.com/drive/19bqVyTF21BlzvEH4wOg_zFoFnCepPH1h?usp=sharing)

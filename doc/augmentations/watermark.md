# WaterMark

The WaterMark augmentation adds watermark effect into input image based on user input word, font, size and etc.


| Parameter                  | Description                                                                            |
|----------------------------|----------------------------------------------------------------------------------------|
| `watermark_word`           | Word for watermark effect.                                                             |
| `watermark_font_size`      | Pair of ints to determine font size of watermark effect.                               |
| `watermark_font_thickness` | Pair of ints to determine thickness of watermark effect.                               |
| `watermark_font_type`      | Font type of watermark effect.                                                         |
| `watermark_rotation`       | Pair of ints to determine angle of rotation in watermark effect.                       |
| `watermark_location`       | Location of watermark effect, select from top, bottom, left, right, center and random. |
| `watermark_color`          | Triplets of ints to determine RGB color of watermark effect.                           |
| `watermark_method`         | Method to overlay watermark foreground into input image.                               |
| `p`                        | The probability that this augmentation will be applied.                                |


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

    watermark= WaterMark(watermark_word = "random",
                         watermark_font_size = (10,15),
                         watermark_font_thickness = (20,25),
                         watermark_font_type = cv2.FONT_HERSHEY_SIMPLEX,
                         watermark_rotation = (0,360) ,
                         watermark_location = "center",
                         watermark_color = (255,0,0),
                         watermark_method = "darken")

    img_watermark = watermark(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input image")
    plt.subplot(122)
    plt.imshow(img_watermark)
    plt.title("WaterMark augmentation")
```

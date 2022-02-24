# Low Ink Periodic Lines

`LowInkPeriodicLines` inherits from `LowInkLine`. LowInkPeriodicLines creates a set of lines that repeat in a periodic fashion throughout the image.


| Parameter           | Description                                                    |
|---------------------|----------------------------------------------------------------|
| `count_range`       | The number of lines to add that will be repeated.              |
| `period_range`      | The number of pixels in each period before lines are repeated. |
| `noise_probability` | The probability to add noise into the generated lines.         |
| `p`                 | The probability that this augmentation will be applied.        |


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
            "Lorem ipsum",
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            0,
            20,
        )


    low_ink_periodic_line_consistent =  LowInkPeriodicLines(
        count_range=(2, 5),
        period_range=(10, 30),
        use_consistent_lines=True,
            )

    low_ink_periodic_line_not_consistent =  LowInkPeriodicLines(
        count_range=(2, 5),
        period_range=(10, 30),
        use_consistent_lines=False,
            )

    img_low_ink_periodic_line_consistent = low_ink_periodic_line_consistent(image)

    img_low_ink_periodic_line_not_consistent = low_ink_periodic_line_not_consistent(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_low_ink_periodic_line_consistent)
    plt.title("Image low ink periodic line consistent")
    plt.figure()
    plt.imshow(img_low_ink_periodic_line_not_consistent)
    plt.title("Image low ink periodic line not consistent")
```


**Example Result:**

![Ink Bleed no Blur](../../images/Augmentations/LowInkLines.png)
![Ink Bleed with Blur](../../images/Augmentations/LowInkLinesBlur.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AfbcC_9_fp8zDnc8GCL0kGApLO67MW7A?usp=sharing)

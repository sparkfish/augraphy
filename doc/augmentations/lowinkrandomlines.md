# Low Ink Random Lines

`LowInkRandomLines` inherits from `LowInkLine`. LowInkRandomLines adds low ink lines randomly throughout the image.


| Parameter     | Description                                             |
|---------------|---------------------------------------------------------|
| `count_range` | The number of lines to add to the image.                |
| `p`           | The probability that this augmentation will be applied. |


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
    

    low_ink_random_line_consistent =  LowInkRandomLines(
        count_range=(20, 40),
        use_consistent_lines=True,
            )
    
    low_ink_random_line_not_consistent =  LowInkRandomLines(
        count_range=(20, 40),
        use_consistent_lines=False,
            )
    
    img_low_ink_random_line_consistent = low_ink_random_line_consistent(image)
    
    img_low_ink_random_line_not_consistent = low_ink_random_line_not_consistent(image)
    
    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")
    
    
    # display output
    plt.figure()
    plt.imshow(img_low_ink_random_line_consistent)
    plt.title("Image low ink random line consistent")
    
    plt.figure()
    plt.imshow(img_low_ink_random_line_not_consistent)
    plt.title("Image low ink random line not consistent")
```


Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1_FX-m7jtfdkucpGL1Ei3Mk_P-mi6DtDq?usp=sharing)



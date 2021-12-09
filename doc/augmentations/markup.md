# Markup

The Markup augmentation uses contours detection to detect text lines and add a smooth text strikethrough, highlight or underline effect.

| Parameter                | Description                                                                     |
|--------------------------|---------------------------------------------------------------------------------|
| `num_lines_range`        | Pair of ints determining the number of added markup effect.                     |
| `markup_length_range`    | Pair of floats between 0 to 1 , to determine the length of added markup effect. |
| `markup_thickness_range` | Pair of ints, to determine the thickness of added markup effect.                |
| `markup_type`            | Choice of markup "strikethrough", "highlight" or "underline".                   |
| `markup_color`           | bgr color tuple.                                                                |
| `p`                      | The probability this augmentation will be applied.                              |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time

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

    # object for strikethrough
    markup_strikethrough = Markup(num_lines_range=(5, 7),
                              markup_length_range=(0.5, 1),
                              markup_thickness_range=(1, 3),
                              markup_type="strikethrough",
                              markup_color=(0, 0, 255))

    # object for highlight
    markup_highlight = Markup(num_lines_range=(5, 7),
                              markup_length_range=(0.5, 1),
                              markup_thickness_range=(1, 3),
                              markup_type="highlight",
                              markup_color=(0, 255, 0))
    # object for underline
    markup_underline = Markup(num_lines_range=(5, 7),
                              markup_length_range=(0.5, 1),
                              markup_thickness_range=(1, 3),
                              markup_type="underline",
                              markup_color=(255, 0, 0))
    
    
    img_strikethrough = markup_strikethrough(image)
    img_highlight = markup_highlight(image)
    img_underline = markup_underline(image)
    
    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_strikethrough)
    plt.title("Strikethrough")
    
    plt.figure()
    plt.imshow(img_highlight)
    plt.title("Highlight")
    
    plt.figure()
    plt.imshow(img_underline)
    plt.title("Underline")

```

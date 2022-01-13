# Letterpress

The Letterpress augmentation produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

| Parameter                | Description                                                                      |
|--------------------------|----------------------------------------------------------------------------------|
| `n_samples`              | Pair of ints determining number of points in a cluster.                          |
| `n_clusters`             | Pair of ints determining number of clusters.                                     |
| `std_range`              | The range from which the standard deviation of the blob distribution is sampled. |
| `value_range`            | The range from which the value of a point in the blob is sampled.                |
| `value_threshold_range`  | Min value of pixel to enable letterpress effect                                  |
| `blur`                   | Flag to enable blur in letterpress noise mask.                                   |
| `p`                      | The probability this augmentation will be applied.                               |


**Example Usage:**
```python
    # Usage Example:

    from matplotlib import pyplot as plt
    from time import time

    # create a blank image
    image = np.full((1500, 1500), 255, dtype="uint8")

    # insert text into image
    for y in range(200, 1300, 200):
        cv2.putText(
            image,
            "Lorem ipsum",
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            0,
            40,
        )

    # create letterpress object
    letterpress = Letterpress(
        n_samples=(300, 400),
        n_clusters=(1000, 1000),
        std_range=(1000, 2000),
        value_range=(230, 255),
    )

    start_time = time()

    # apply letterpress
    image_output = letterpress(image)

    elapsed_time = time() - start_time

    # processing time per image
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(image_output, cmap="gray")

```

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1qp_ZnbdP0A4eIhxkF0h7Dduw_9gtAhti?usp=sharing)

# Dirty Drum

The Dirty Drum augmentation emulates deposits of dirt and ink-grime from dirty printer drums.

**Example Usage:**

```python
augmentation = DirtyDrum(
	line_width_range=(2, 8),
	noise_intensity= 1,
	ksize=(3, 3),
	sigmaX=0
	)
```

| Parameter          | Description                                                     |
|--------------------|-----------------------------------------------------------------|
| `line_width_range` | The range from which the width of a dirty drum line is sampled. |
| `direction`        | Direction of effect, 0=horizontal, 1=vertical, 2=both.          |
| `noise_intensity`  | Intensity of the effect.                                        |
| `ksize`            | Height/width pair from which to sample the kernel size.         |
| `sigmaX`           | Standard deviation of the kernel along the x-axis.              |
| `p`                | The probability this augmentation will be applied.              |


```python
    # Usage Example:

    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

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

    # create dirtydrum object
    dirtydrum1 = DirtyDrum(
            line_width_range=(2, 8),
			direction = 1,
            noise_intensity= 1,
            ksize=(3, 3),
            sigmaX=0
            )

    start_time = time()

    # apply dirty drum
    image_output = dirtydrum1(image)

    elapsed_time = time() - start_time

    # processing time per image
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    plt.figure()
    plt.imshow(image_output, cmap="gray")

```


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yl0AxrAE2RyWnkWh07yWRDZdHJVfIfqb?usp=sharing)

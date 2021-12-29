# Noise Texturize

The Noise Texturize augmentation creates a random noise based texture pattern to emulate paper textures.


| Parameter          | Description                                                                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sigma_range`      | The bounds of noise fluctuations.                                                                                                                    |
| `turbulence_range` | How quickly big patterns will be replaced with the small ones. The lower the value, the more iterations will be performed during texture generation. |
| `p`                | The probability that this augmentation will be applied.                                                                                              |


**Example Usage:**
```python
    from matplotlib import pyplot as plt
    from time import time
    import cv2
    import numpy as np

    start_time = time()

    # create a blank image
    image = np.full((1500, 1500, 3), 255, dtype="uint8")

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


    noise_texturize = NoiseTexturize(
            sigma_range=(12, 15),
            turbulence_range=(3, 5),
            )

    img_noise_texturize = noise_texturize(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_noise_texturize)
    plt.title("Noise texturized image")
```


**Example Result:**

![Noise textureized image](../../images/Augmentations/NoiseTexturizeAfter.png)

Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1UOwbY83BeEbpRXiSQlhZvodUvrUVPpXh?usp=sharing)

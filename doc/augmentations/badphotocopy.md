# Bad Photocopy

The Bad Photocopy augmentation produces noise blobs simulating a dirty copier.

| Parameter            | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `mask`               | Mask of noise to generate badphotocopy effect.                             |
| `noise_type`         | Types of noises to generate different mask patterns.                       |
| `noise_side`         | Location of noise.                                                         |
| `noise_iteration`    | Pair of ints to determine number of iterations to apply noise in the mask. |
| `noise_size`         | Pair of ints to determine scale of noise in the mask.                      |
| `noise_value`        | Intensity range of noise, lower value get darker effect.                   |
| `noise_sparsity`	   | Pair of floats determining sparseness of noise.                            |
| `noise_concentration`| Pair of floats determining concentration of noise.                         |
| `blur_noise`         | Flag to enable blur in noise mask.                                         |
| `blur_noise_kernel`  | Kernel to blur noise mask.                                                 |
| `wave_pattern`       | To enable wave pattern in noise.                                           |
| `edge_effect`        | To add sobel edge effect into the noise mask.                              |
| `p`                  | The probability this augmentation will be applied.                         |


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

    BadPhotoCopy_type_1 = BadPhotoCopy(noise_type=1,
                                       noise_side="left",
                                       noise_iteration=(2,3),
                                       noise_size=(2,3),
                                       noise_sparsity=(0.2,0.3),
                                       noise_concentration=(0.05,0.1))

    BadPhotoCopy_type_2 = BadPhotoCopy(noise_type=2,
                                       noise_side="right",
                                       noise_iteration=(2,3),
                                       noise_size=(2,3),
                                       noise_sparsity=(0.4,0.5),
                                       noise_concentration=(0.1,0.2))

    BadPhotoCopy_type_3 = BadPhotoCopy(noise_type=3,
                                       noise_iteration=(2,3),
                                       noise_size=(1,3),
                                       noise_sparsity=(0.2,0.3),
                                       noise_concentration=(0.5,0.5))

    BadPhotoCopy_type_4 = BadPhotoCopy(noise_type=4,
                                       noise_iteration=(2,3),
                                       noise_size=(1,3),
                                       noise_sparsity=(0.5,0.5),
                                       noise_concentration=(0.99,0.99))

    img_BadPhotoCopy_type_1 = BadPhotoCopy_type_1(image)
    img_BadPhotoCopy_type_2 = BadPhotoCopy_type_2(image)
    img_BadPhotoCopy_type_3 = BadPhotoCopy_type_3(image)
    img_BadPhotoCopy_type_4 = BadPhotoCopy_type_4(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    for i in range(4):
        plt.figure()
        eval("plt.imshow(img_BadPhotoCopy_type_" + str(i + 1) + ", cmap='gray')")
        plt.title("BadPhotoCopy type " + str(i + 1))

```
**Example Result:**

![Original Image](../../images/Augmentations/BadPhotoCopyOriginal.png)
![Bad Photo Copy 1](../../images/Augmentations/BadPhotoCopyExample1.png)
![Bad Photo Copy 2](../../images/Augmentations/BadPhotoCopyExample2.png)
![Bad Photo Copy 3](../../images/Augmentations/BadPhotoCopyExample3.png)
![Bad Photo Copy 4](../../images/Augmentations/BadPhotoCopyExample4.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gdo3mI65umZ6lTTqVIZJTE6YPogIBOzM?usp=sharing)

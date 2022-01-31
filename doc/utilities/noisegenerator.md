# NoiseGenerator

Core object to generate mask of noise.

| Parameter             | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `noise_type`          | Types of generated noise.                                        |
| `noise_side`          | Location of noise.                                               |
| `noise_value`         | Pair of ints determining value of noise.                         |
| `noise_background`    | Pair of ints determining value of noise background.              |
| `noise_sparsity`      | Pair of floats determining sparseness of noise                   |
| `noise_concentration` | Pair of floats determining concentration of noise.               |
| `xsize`               | Number of columns in generated mask of noise.                    |
| `ysize`               | Number of rows in generated mask of noise.                       |

**Example Usage:**
```python

    from matplotlib import pyplot as plt
    from time import time

    start_time = time()

    # initialize noise generator
    noise_generator_type_1_topleft = NoiseGenerator(noise_type=1, noise_side="top_left")
    noise_generator_type_1_topright = NoiseGenerator(noise_type=1, noise_side="top_right")
    noise_generator_type_1_bottomleft = NoiseGenerator(noise_type=1, noise_side="bottom_left")
    noise_generator_type_1_bottomright = NoiseGenerator(noise_type=1, noise_side="bottom_right")

    noise_generator_type_2_left = NoiseGenerator(noise_type=2, noise_side="left")
    noise_generator_type_2_right = NoiseGenerator(noise_type=2, noise_side="right")
    noise_generator_type_2_top = NoiseGenerator(noise_type=2, noise_side="top")
    noise_generator_type_2_bottom = NoiseGenerator(noise_type=2, noise_side="bottom")

    noise_generator_type_3 = NoiseGenerator(noise_type=3)

    noise_generator_type_4 = NoiseGenerator(noise_type=4)

    # generate mask of noise
    img_mask_type_1_topleft = noise_generator_type_1_topleft.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.2, 0.3),
    )
    img_mask_type_1_topright = noise_generator_type_1_topright.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.2, 0.3),
    )
    img_mask_type_1_bottomleft = noise_generator_type_1_bottomleft.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.2, 0.3),
    )
    img_mask_type_1_bottomright = noise_generator_type_1_bottomright.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.2, 0.3),
    )

    img_mask_type_2_left = noise_generator_type_2_left.generate_noise(
        noise_iteration=(1, 1),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.5),
        noise_concentration=(0.05, 0.1),
    )

    img_mask_type_2_right = noise_generator_type_2_right.generate_noise(
        noise_iteration=(1, 1),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.5),
        noise_concentration=(0.05, 0.1),
    )

    img_mask_type_2_top = noise_generator_type_2_top.generate_noise(
        noise_iteration=(1, 1),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.5),
        noise_concentration=(0.05, 0.1),
    )

    img_mask_type_2_bottom = noise_generator_type_2_bottom.generate_noise(
        noise_iteration=(1, 1),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.5),
        noise_concentration=(0.05, 0.1),
    )

    img_mask_type_3 = noise_generator_type_3.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.2, 0.3),
    )

    img_mask_type_4 = noise_generator_type_4.generate_noise(
        noise_iteration=(2, 3),
        noise_size=(2, 3),
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.2, 0.3),
        noise_concentration=(0.6, 0.9),
    )

    elapsed_time = time() - start_time

    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display outputs
    plt.figure()
    plt.subplot(221)
    plt.imshow(img_mask_type_1_topleft)
    plt.title("noise type 1 - top left")
    plt.subplot(222)
    plt.imshow(img_mask_type_1_topright)
    plt.title("noise type 1 - top right")
    plt.subplot(223)
    plt.imshow(img_mask_type_1_bottomleft)
    plt.title("noise type 1 - bottom left")
    plt.subplot(224)
    plt.imshow(img_mask_type_1_bottomright)
    plt.title("noise type 1 - bottom right")

    plt.figure()
    plt.subplot(221)
    plt.imshow(img_mask_type_2_left)
    plt.title("noise type 2 - left")
    plt.subplot(222)
    plt.imshow(img_mask_type_2_right)
    plt.title("noise type 2 - right")
    plt.subplot(223)
    plt.imshow(img_mask_type_2_top)
    plt.title("noise type 2 - top")
    plt.subplot(224)
    plt.imshow(img_mask_type_2_bottom)
    plt.title("noise type 2 - bottom")

    plt.figure()
    plt.imshow(img_mask_type_3)
    plt.title("noise type 3")

    plt.figure()
    plt.imshow(img_mask_type_4)
    plt.title("noise type 4")

```

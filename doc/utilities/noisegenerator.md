# NoiseGenerator

Core object to generate mask of noise.

| Parameter             | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `noise_type`          | Types of generated noise.                                        |
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
    noise_generator_type_1 = NoiseGenerator(noise_type=1)
    noise_generator_type_2 = NoiseGenerator(noise_type=2)
    noise_generator_type_3 = NoiseGenerator(noise_type=3)
    noise_generator_type_4 = NoiseGenerator(noise_type=4)
    noise_generator_type_5 = NoiseGenerator(noise_type=5)
    noise_generator_type_6 = NoiseGenerator(noise_type=6)
    noise_generator_type_7 = NoiseGenerator(noise_type=7)
    noise_generator_type_8 = NoiseGenerator(noise_type=8)

    # generate mask of noise
    img_mask_type_1 = noise_generator_type_1.generate_noise(
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.6, 0.7),
        noise_concentration=(0.7, 0.8),
    )
    img_mask_type_2 = noise_generator_type_2.generate_noise(
        noise_value=(0, 64),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.7, 0.8),
    )
    img_mask_type_3 = noise_generator_type_3.generate_noise(
        noise_value=(0, 128),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.5),
        noise_concentration=(0.5, 0.5),
    )
    img_mask_type_4 = noise_generator_type_4.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.9, 0.9),
        noise_concentration=(0.9, 0.9),
    )
    img_mask_type_5 = noise_generator_type_5.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_6 = noise_generator_type_6.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_7 = noise_generator_type_7.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )
    img_mask_type_8 = noise_generator_type_8.generate_noise(
        noise_value=(0, 50),
        noise_background=(255, 255),
        noise_sparsity=(0.5, 0.6),
        noise_concentration=(0.5, 0.6),
    )

    elapsed_time = time() - start_time

    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")

    # display output
    for i in range(8):
        plt.figure()
        eval("plt.imshow(img_mask_type_" + str(i + 1) + ", cmap='gray')")
        plt.title("noise mask type " + str(i + 1))

```

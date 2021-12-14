# Bindings And Fasteners

The Bindings And Fasteners augmentation creates binding and fastener mark in the input image.


| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `overlay_types` | Types of overlay method, min, max or mix.                        |
| `foreground`    | Path to foreground image.                                        |
| `effect_type`   | Types of binding effect.                                         |
| `ntimes`        | Number of repetition to draw foreground image.                   |
| `edge`          | Which edge of the page the foreground copies should be placed on.|
| `edge_offset`   | How far from the edge of the page to draw the copies.            |
| `p`             | The probability this augmentation will be applied.               |


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

    # create a foreground image
    image_foreground = np.full((100, 150,3), 255, dtype="uint8")
    # insert text into foreground
    cv2.putText(
            image_foreground,
            "APPROVE",
            (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3,
        )


    binder_punch_holes = BindingsAndFasteners(overlay_types="darken",
                                        foreground=None,
                                        effect_type="punch_holes",
                                        ntimes=3,
                                        nscales=(1, 1),
                                        edge="left",
                                        edge_offset=50,
                                    )

    binder_binding_holes = BindingsAndFasteners(overlay_types="darken",
                                        foreground=None,
                                        effect_type="binding_holes",
                                        ntimes=10,
                                        nscales=(2, 2),
                                        edge="right",
                                        edge_offset=50,
                                    )

    binder_clips = BindingsAndFasteners(overlay_types="darken",
                                        foreground=None,
                                        effect_type="clips",
                                        ntimes=3,
                                        nscales=(3, 4),
                                        edge="bottom",
                                        edge_offset=50,
                                    )

    binder_input_foreground = BindingsAndFasteners(overlay_types="darken",
                                        foreground=image_foreground,
                                        ntimes=3,
                                        nscales=(1, 1),
                                        edge="random",
                                    )

    img_punch_holes = binder_punch_holes(image)
    img_binding_holes = binder_binding_holes(image)
    img_clips = binder_clips(image)
    img_input_foreground  = binder_input_foreground(image)

    elapsed_time = time() - start_time
    # processing time
    print("Elapsed time = " + str(elapsed_time) + " seconds")


    # display output
    plt.figure()
    plt.imshow(img_punch_holes)
    plt.title("Punch holes")

    plt.figure()
    plt.imshow(img_binding_holes)
    plt.title("binding holes")

    plt.figure()
    plt.imshow(img_clips)
    plt.title("Clips")

    plt.figure()
    plt.imshow(img_input_foreground)
    plt.title("Using foreground")

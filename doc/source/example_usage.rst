=============
Example Usage
=============

Augraphy uses pipeline in the augmentation process so an augmentation pipeline need to be initialized prior the augmentation process. The resulting output is a dictionary containing all the outputs parameters.

------------------------------------------------------------
Examples 1 : Augmentation with default augmentation pipeline
------------------------------------------------------------

To use the default pipeline which contains all available augmentations and sensible defaults::

    from augraphy import *

    pipeline = default_augraphy_pipeline()

    img = cv2.imread("image.png")

    data = pipeline.augment(img)

    augmented = data["output"]

------------------------------------------------------------
Examples 2 : Augmentation with custom augmentation pipeline
------------------------------------------------------------

To initialize a custom augmentation pipeline and augment an image::

    from augraphy import *
    import random

    ink_phase = [

        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 16),
            kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
            severity=(0.4, 0.6),
            p=0.33,
        ),
        OneOf(
            [
                Letterpress(
                    n_samples=(100, 400),
                    n_clusters=(200, 400),
                    std_range=(500, 3000),
                    value_range=(150, 224),
                    value_threshold_range=(96, 128),
                    blur=1,
                ),
                BleedThrough(
                    intensity_range=(0.1, 0.3),
                    color_range=(32, 224),
                    ksize=(17, 17),
                    sigmaX=1,
                    alpha=random.uniform(0.1, 0.2),
                    offsets=(10, 20),
                ),
            ],
        ),
    ]

    paper_phase = [
        PaperFactory(p=0.33),
        ColorPaper(
            hue_range=(0, 255),
            saturation_range=(10, 40),
            p=0.33,
        ),
        WaterMark(
            watermark_word="random",
            watermark_font_size=(10, 15),
            watermark_font_thickness=(20, 25),
            watermark_rotation=(0, 360),
            watermark_location="random",
            watermark_color="random",
            watermark_method="darken",
            p=0.33,
        ),
        AugmentationSequence(
            [
                NoiseTexturize(
                    sigma_range=(3, 10),
                    turbulence_range=(2, 5),
                ),
                BrightnessTexturize(
                    brightness_range=(0.9, 0.99),
                    deviation=0.03,
                ),
            ],
        ),
    ]

    post_phase = [

        DirtyRollers(
            line_width_range=(2, 32),
            scanline_type=0,
            p=0.33,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.33,
        ),
        SubtleNoise(
            subtle_range=random.randint(5, 10),
            p=0.33,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.33,
        ),
        Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=random.randint(1, 6),
            fold_noise=random.uniform(0, 0.2),
            gradient_width=(0.1, 0.2),
            gradient_height=(0.01, 0.02),
            p=0.33,
        ),
        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(1, 2),
            markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
            markup_color="random",
            single_word_mode=False,
            repetitions=1,
            p=0.33,
        ),
        PencilScribbles(
            size_range=(100, 800),
            count_range=(1, 6),
            stroke_count_range=(1, 2),
            thickness_range=(2, 6),
            brightness_change=random.randint(64, 224),
            p=0.33,
        ),
        BadPhotoCopy(
            mask=None,
            noise_type=-1,
            noise_side="random",
            noise_iteration=(1, 2),
            noise_size=(1, 3),
            noise_value=(128, 196),
            noise_sparsity=(0.3, 0.6),
            noise_concentration=(0.1, 0.6),
            blur_noise=random.choice([True, False]),
            blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
            wave_pattern=random.choice([True, False]),
            edge_effect=random.choice([True, False]),
            p=0.33,
        ),
        Faxify(
            scale_range=(0.3, 0.6),
            monochrome=random.choice([0, 1]),
            monochrome_method="random",
            monochrome_arguments={},
            halftone=random.choice([0, 1]),
            invert=1,
            half_kernel_size=random.choice([(1, 1), (2, 2)]),
            angle=(0, 360),
            sigma=(1, 3),
            p=0.33,
        ),
        BookBinding(
            radius_range=(1, 100),
            curve_range=(100, 200),
            mirror_range=(0.3, 0.5),
            p=0.33,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    img = cv2.imread("image.png")

    data = pipeline.augment(img)

    augmented = data["output"]

"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline():

    ink_phase = [
        Dithering(
            dither=random.choice(["ordered", "floyd-steinberg"]),
            order=(3, 5),
            p=0.33,
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 16),
            kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
            severity=(0.4, 0.6),
            p=0.33,
        ),
        BleedThrough(
            intensity_range=(0.1, 0.3),
            color_range=(32, 224),
            ksize=(17, 17),
            sigmaX=1,
            alpha=random.uniform(0.1, 0.2),
            offsets=(10, 20),
            p=0.33,
        ),
        Letterpress(
            n_samples=(100, 400),
            n_clusters=(200, 400),
            std_range=(500, 3000),
            value_range=(150, 224),
            value_threshold_range=(96, 128),
            blur=1,
            p=0.33,
        ),
        OneOf(
            [
                LowInkRandomLines(
                    count_range=(5, 10),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
                ),
                LowInkPeriodicLines(
                    count_range=(2, 5),
                    period_range=(16, 32),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
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
        OneOf(
            [
                AugmentationSequence(
                    [
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                    ],
                ),
                AugmentationSequence(
                    [
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                    ],
                ),
            ],
            p=0.33,
        ),
        Brightness(
            brightness_range=(0.9, 1.1),
            min_brightness=0,
            min_brightness_value=(120, 150),
            p=0.1,
        ),
    ]

    post_phase = [
        OneOf(
            [
                PageBorder(
                    side="random",
                    border_background_value=(230, 255),
                    flip_border=random.choice([0, 1]),
                    width_range=(5, 30),
                    pages=None,
                    noise_intensity_range=(0.3, 0.8),
                    curve_frequency=(2, 8),
                    curve_height=(2, 4),
                    curve_length_one_side=(50, 100),
                    value=(32, 150),
                    same_page_border=random.choice([0, 1]),
                ),
                DirtyRollers(
                    line_width_range=(2, 32),
                    scanline_type=0,
                ),
            ],
            p=0.33,
        ),
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
            ],
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
        Gamma(
            gamma_range=(0.9, 1.1),
            p=0.33,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="random",
            ntimes=(2, 6),
            nscales=(0.9, 1.0),
            edge="random",
            edge_offset=(10, 50),
            use_figshare_library=0,
            p=0.33,
        ),
        Geometric(
            scale=(0.75, 1.25),
            translation=(-10, 10),
            fliplr=random.choice([True, False]),
            flipud=random.choice([True, False]),
            crop=(),
            rotate_range=(-5, 5),
            randomize=0,
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

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase, log=False)

    return pipeline


def default_augment(img):

    default_pipeline = default_augraphy_pipeline()

    return default_pipeline.augment(img)


def pipeline_archetype1():

    ink_phase = [
        Geometric(
            padding=(0, 0, 0.05, 0),
            randomize=0,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            effect_type="punch_holes",
            ntimes=(3, 3),
            nscales=(1.0, 1.0),
            edge="top",
            edge_offset=(0.01, 0.01),
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 5),
            kernel_size=(3, 3),
            severity=(0.2, 0.2),
        ),
    ]

    paper_phase = [
        PageBorder(
            side="right",
            border_background_value=(0, 0),
            pages=1,
            width_range=(2, 3),
            same_page_border=0,
        ),
        Geometric(
            translation=(-0.1, 0.2),
            randomize=0,
        ),
        Geometric(
            translation=(0, -0.05),
            randomize=0,
        ),
    ]

    post_phase = [
        Geometric(
            rotate_range=(-1, -1),
            randomize=0,
        ),
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
        Brightness(
            brightness_range=(2.0, 2.0),
            min_brightness=1,
            min_brightness_value=(50, 50),
        ),
        Gamma(
            gamma_range=(1.5, 1.5),
        ),
        Letterpress(
            n_samples=(1000, 1000),
            n_clusters=(500, 500),
            std_range=(400, 400),
            value_range=(200, 255),
            value_threshold_range=(255, 255),
            blur=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="bottom",
            noise_iteration=(1, 1),
            noise_size=(5, 7),
            noise_value=(0, 64),
            noise_sparsity=(0.01, 0.01),
            noise_concentration=(0.01, 0.01),
            blur_noise=1,
            wave_pattern=0,
            edge_effect=0,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_side="random",
            noise_iteration=(4, 5),
            noise_size=(1, 4),
            noise_value=(0, 64),
            noise_sparsity=(0.5, 0.6),
            noise_concentration=(0.5, 0.5),
            blur_noise=1,
            wave_pattern=0,
            edge_effect=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline


def pipeline_archetype2():

    ink_phase = [
        Markup(
            num_lines_range=(3, 4),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(3, 5),
            markup_type="underline",
            markup_color=(0, 0, 0),
            single_word_mode=False,
            large_word_mode=False,
            repetitions=1,
        ),
        InkBleed(
            intensity_range=(0.9, 0.9),
            color_range=(0, 0),
            kernel_size=(3, 3),
            severity=(0.1, 0.2),
        ),
    ]

    paper_phase = [
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="punch_holes",
            ntimes=(3, 3),
            nscales=(1, 1),
            edge="left",
            edge_offset=(40, 40),
        ),
        PageBorder(
            side="right",
            noise_intensity_range=(0.5, 0.8),
            width_range=(5, 10),
        ),
        PageBorder(
            side="top",
            noise_intensity_range=(0.5, 0.8),
            width_range=(3, 6),
        ),
        PageBorder(
            side="left",
            noise_intensity_range=(0.5, 0.8),
            width_range=(1, 2),
        ),
        PageBorder(
            side="bottom",
            noise_intensity_range=(0.5, 0.8),
            width_range=(1, 1),
        ),
    ]

    post_phase = [
        BadPhotoCopy(
            noise_type=2,
            noise_side="top",
            noise_iteration=(2, 3),
            noise_size=(1, 1),
            noise_value=(15, 30),
            noise_sparsity=(0.2, 0.3),
            noise_concentration=(0.05, 0.05),
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="bottom",
            noise_value=(0, 15),
            noise_sparsity=(0.025, 0.025),
            noise_concentration=(0.015, 0.015),
            wave_pattern=0,
            edge_effect=0,
        ),
        Geometric(rotate_range=(2, 2), randomize=0),
        Geometric(padding=(0.1, 0.2, 0.2, 0.05), randomize=0),
        BadPhotoCopy(
            noise_type=2,
            noise_side="left",
            noise_value=(0, 15),
            noise_sparsity=(0.025, 0.025),
            noise_concentration=(0.025, 0.025),
            wave_pattern=0,
            edge_effect=0,
        ),
        PencilScribbles(
            size_range=(200, 300),
            count_range=(2, 2),
            stroke_count_range=(1, 1),
            thickness_range=(2, 6),
            brightness_change=128,
        ),
        BindingsAndFasteners(
            foreground=None,
            overlay_types="darken",
            effect_type="clips",
            ntimes=(3, 3),
            nscales=(1, 1),
            edge="bottom",
            edge_offset=(10, 10),
        ),
        BindingsAndFasteners(
            foreground=None,
            overlay_types="darken",
            effect_type="binding_holes",
            ntimes=(5, 5),
            nscales=(1, 1),
            edge="right",
            edge_offset=(10, 10),
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="top",
            noise_value=(0, 15),
            noise_sparsity=(0.7, 0.7),
            noise_concentration=(0.05, 0.05),
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(2, 4),
            noise_size=(1, 2),
            noise_value=(0, 5),
            noise_sparsity=(0.7, 0.7),
            noise_concentration=(0.1, 0.1),
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="right",
            noise_value=(0, 15),
            noise_sparsity=(0.025, 0.025),
            noise_concentration=(0.015, 0.015),
            wave_pattern=0,
            edge_effect=0,
        ),
        Geometric(padding=(0.01, 0.01, 0.01, 0), randomize=0),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline


def pipeline_archetype3():

    ink_phase = [
        InkBleed(
            intensity_range=(0.9, 0.9),
            color_range=(0, 0),
            kernel_size=(7, 7),
            severity=(0.3, 0.3),
        ),
    ]

    paper_phase = [
        BadPhotoCopy(
            noise_type=3,
            noise_side="top",
            noise_value=(110, 110),
            noise_sparsity=(0.0025, 0.0025),
            noise_concentration=(0.3, 0.4),
            blur_noise=1,
            blur_noise_kernel=(31, 31),
            edge_effect=0,
            wave_pattern=0,
        ),
        BadPhotoCopy(
            noise_type=3,
            noise_side="top",
            noise_iteration=(1, 2),
            noise_size=(1, 1),
            noise_value=(140, 150),
            noise_sparsity=(0.1, 0.1),
            noise_concentration=(0.1, 0.2),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
    ]

    post_phase = [
        BadPhotoCopy(
            noise_type=2,
            noise_side="right",
            noise_iteration=(4, 5),
            noise_size=(5, 7),
            noise_value=(100, 120),
            noise_sparsity=(0.4, 0.5),
            noise_concentration=(0.05, 0.05),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(3, 4),
            noise_size=(3, 5),
            noise_value=(15, 39),
            noise_sparsity=(0.3, 0.6),
            noise_concentration=(0.15, 0.25),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
        LightingGradient(
            light_position=(800, 2000),
            direction=0,
            max_brightness=255,
            min_brightness=25,
            mode="gaussian",
            transparency=0.75,
        ),
        BleedThrough(
            intensity_range=(0.01, 0.02),
            color_range=(0, 10),
            ksize=(7, 7),
            sigmaX=0,
            alpha=0.2,
            offsets=(10, 20),
        ),
        Geometric(rotate_range=(2, 2), randomize=0),
        SubtleNoise(p=1),
        Jpeg(p=1),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline


def pipeline_archetype4():

    ink_phase = [
        Geometric(
            crop=(0.12, 0.08, 0.85, 0.9),
            randomize=0,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            effect_type="punch_holes",
            ntimes=(3, 3),
            nscales=(1.0, 1.0),
            edge="left",
            edge_offset=(10, 10),
        ),
        InkBleed(
            intensity_range=(0.5, 0.5),
            color_range=(0, 0),
            kernel_size=(3, 3),
            severity=(0.5, 0.5),
        ),
    ]

    paper_phase = [
        PageBorder(
            side="top",
            border_background_value=(0, 0),
            pages=1,
            width_range=(0.03, 0.03),
            same_page_border=0,
        ),
        PageBorder(
            side="bottom",
            border_background_value=(0, 0),
            pages=1,
            width_range=(0.05, 0.05),
            same_page_border=0,
        ),
    ]

    post_phase = [
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
        Geometric(
            rotate_range=(-1, -1),
            padding=(0, 0.05, 0, 0),
            randomize=0,
        ),
        Geometric(
            crop=(0, 0.02, 1, 0.99),
            randomize=0,
        ),
        BadPhotoCopy(
            noise_type=4,
            noise_side="random",
            noise_iteration=(2, 3),
            noise_size=(5, 7),
            noise_value=(0, 64),
            noise_sparsity=(0.2, 0.3),
            noise_concentration=(0.05, 0.05),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="top_left",
            noise_iteration=(1, 1),
            noise_size=(9, 11),
            noise_value=(0, 64),
            noise_sparsity=(0.01, 0.01),
            noise_concentration=(0.02, 0.02),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline


def pipeline_archetype5():

    ink_phase = [
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
        InkBleed(
            intensity_range=(0.3, 0.4),
            color_range=(0, 0),
            kernel_size=(3, 3),
            severity=(1.0, 1.0),
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="right",
            noise_iteration=(30, 30),
            noise_size=(1, 4),
            noise_value=(0, 1),
            noise_sparsity=(0.7, 0.7),
            noise_concentration=(0.025, 0.025),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="left",
            noise_iteration=(30, 30),
            noise_size=(1, 4),
            noise_value=(0, 1),
            noise_sparsity=(0.1, 0.1),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=1,
        ),
        Geometric(randomize=0, padding=(0, 0.02, 0, 0)),
    ]

    paper_phase = [
        PageBorder(
            side="right",
            noise_intensity_range=(0.5, 0.8),
            width_range=(4, 5),
        ),
        PageBorder(
            side="bottom",
            noise_intensity_range=(0.5, 0.8),
            width_range=(4, 5),
        ),
    ]

    post_phase = [
        Geometric(randomize=0, rotate_range=(-2, -2)),
        Geometric(randomize=0, translation=(0.02, -0.05)),
        PageBorder(
            side="left",
            width_range=(6, 7),
            pages=5,
            noise_intensity_range=(0.0, 0.2),
        ),
        PageBorder(
            side="bottom",
            width_range=(8, 12),
            pages=5,
            noise_intensity_range=(0.0, 0.2),
        ),
        BindingsAndFasteners(
            overlay_types="min",
            effect_type="clips",
            ntimes=(2, 2),
            nscales=(1.5, 1.5),
            edge="bottom",
            edge_offset=(20, 20),
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline


def pipeline_archetype9():

    ink_phase = [
        Letterpress(
            n_samples=(200, 300),
            n_clusters=(500, 680),
            std_range=(2500, 2500),
            value_range=(245, 255),
            value_threshold_range=(128, 128),
            blur=0,
        ),
        Geometric(scale=(2, 2), randomize=0),
        Faxify(
            scale_range=(1.0, 1.0),
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
            invert=1,
            half_kernel_size=(1, 1),
            angle=(73, 73),
            sigma=(3, 3),
        ),
        Dithering(dither="ordered", order=(2, 2)),
        Geometric(scale=(0.5, 0.5), randomize=0),
        Markup(
            num_lines_range=(1, 1),
            markup_length_range=(0.4, 0.4),
            markup_thickness_range=(2, 3),
            markup_type="underline",
            markup_color=(0, 0, 0),
            single_word_mode=False,
            large_word_mode=False,
        ),
        Geometric(translation=(0, 0.15), randomize=0),
    ]

    paper_phase = [
        DirtyDrum(
            line_width_range=(1, 1),
            line_concentration=0.15,
            direction=0,
            noise_intensity=0.15,
            noise_value=(0, 15),
            ksize=(3, 3),
        ),
        Geometric(translation=(0, 0.9), randomize=0),
        BadPhotoCopy(
            noise_type=4,
            noise_iteration=(2, 3),
            noise_size=(1, 3),
            noise_value=(0, 1),
            noise_sparsity=(0.2, 0.3),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
    ]
    post_phase = [
        Geometric(rotate_range=(-1, -1), randomize=0),
    ]

    pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    return pipeline

"""The default recommended pipeline.

If you don't need to produce your own augmentations or specialized pipelines,
you can use this to generate more images.
"""
import random

from augraphy import *


def default_augraphy_pipeline():

    pre_phase = [
        # Rescale(scale="optimal", target_dpi = 300,  p = 1.0),
    ]

    ink_phase = [
        Dithering(
            dither=random.choice(["ordered", "floyd-steinberg"]),
            order=(3, 5),
            p=0.33,
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
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
                            texture_width_range=(300, 500),
                            texture_height_range=(300, 500),
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
                            texture_width_range=(300, 500),
                            texture_height_range=(300, 500),
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
                    page_border_width_height="random",
                    page_border_color=(0, 0, 0),
                    page_border_background_color=(0, 0, 0),
                    page_numbers="random",
                    page_rotation_angle_range=(-3, 3),
                    curve_frequency=(2, 8),
                    curve_height=(2, 4),
                    curve_length_one_side=(50, 100),
                    same_page_border=random.choice([0, 1]),
                    numba_jit=1,
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
            fold_count=random.randint(5, 10),
            fold_noise=random.uniform(0, 0.01),
            fold_angle_range=(-360, 360),
            gradient_width=(0.1, 0.2),
            gradient_height=(0.01, 0.02),
            backdrop_color=(0, 0, 0),
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
        Scribbles(
            scribbles_type="random",
            scribbles_location="random",
            scribbles_size_range=(250, 600),
            scribbles_count_range=(1, 6),
            scribbles_thickness_range=(1, 3),
            scribbles_brightness_change=[32, 64, 128],
            scribbles_text="random",
            scribbles_text_font="random",
            scribbles_text_rotate_range=(0, 360),
            scribbles_lines_stroke_count_range=(1, 6),
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
            width_range="random",
            height_range="random",
            angle_range=(-30, 30),
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
            shadow_radius_range=(1, 100),
            curve_range_right=(100, 200),
            mirror_range=(0.3, 0.5),
            p=0.33,
        ),
        # Rescale(scale = "original" , p = 1.0)
    ]

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        log=False,
    )

    return pipeline


def default_augment(img):

    default_pipeline = default_augraphy_pipeline()

    return default_pipeline.augment(img)


def pipeline_archetype1():

    ink_phase = [
        Geometric(
            padding=[0, 0, 0.05, 0],
            randomize=0,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            effect_type="punch_holes",
            width_range="random",
            height_range="random",
            ntimes=(3, 3),
            nscales=(1.0, 1.0),
            edge="top",
            edge_offset=(0.01, 0.01),
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            kernel_size=(3, 3),
            severity=(0.2, 0.2),
        ),
    ]

    paper_phase = [
        PageBorder(
            page_border_width_height=(0, 2),
            page_border_color=(0, 0, 0),
            page_border_background_color=(0, 0, 0),
            page_numbers=1,
            page_rotation_angle_range=(0, 0),
            curve_frequency=(2, 8),
            curve_height=(2, 4),
            curve_length_one_side=(50, 100),
            same_page_border=0,
        ),
        Scribbles(
            scribbles_type="text",
            scribbles_ink="pen",
            scribbles_location=(0.9, 0.75),
            scribbles_size_range=(700, 700),
            scribbles_count_range=(1, 1),
            scribbles_thickness_range=(1, 1),
            scribbles_brightness_change=[96],
            scribbles_color=(0, 0, 0),
            scribbles_text="600",
            scribbles_text_font="random",
            scribbles_text_rotate_range=(30, 30),
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
        Scribbles(
            scribbles_type="text",
            scribbles_ink="pencil",
            scribbles_location=(0.5, 0.95),
            scribbles_size_range=(700, 700),
            scribbles_count_range=(1, 1),
            scribbles_thickness_range=(1, 1),
            scribbles_brightness_change=[0],
            scribbles_skeletonize=0,
            scribbles_color=(0, 0, 0),
            scribbles_text="P·142",
            scribbles_text_font="random",
            scribbles_text_rotate_range=(3, 3),
        ),
        BadPhotoCopy(
            noise_type=5,
            noise_side="bottom",
            noise_iteration=(1, 1),
            noise_size=(5, 7),
            noise_value=(64, 128),
            noise_sparsity=(0.2, 0.2),
            noise_concentration=(0.2, 0.2),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(2, 3),
            noise_size=(1, 4),
            noise_value=(0, 64),
            noise_sparsity=(0.5, 0.6),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

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
            kernel_size=(3, 3),
            severity=(0.1, 0.2),
        ),
    ]

    paper_phase = [
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="punch_holes",
            width_range="random",
            height_range="random",
            ntimes=(3, 3),
            nscales=(1, 1),
            edge="left",
            edge_offset=(40, 40),
        ),
        PageBorder(
            page_border_width_height=(5, 0),
            page_border_color=(0, 0, 0),
            page_border_background_color=(0, 0, 0),
            page_numbers=2,
            page_rotation_angle_range=(0, 0),
        ),
        PageBorder(
            page_border_width_height=(0, -5),
            page_border_color=(0, 0, 0),
            page_border_background_color=(0, 0, 0),
            page_numbers=1,
            page_rotation_angle_range=(0, 0),
        ),
        PageBorder(
            page_border_width_height=(0, 1),
            page_border_color=(0, 0, 0),
            page_border_background_color=(0, 0, 0),
            page_numbers=1,
            page_rotation_angle_range=(0, 0),
        ),
        PageBorder(
            page_border_width_height=(-1, 0),
            page_border_color=(0, 0, 0),
            page_border_background_color=(0, 0, 0),
            page_numbers=1,
            page_rotation_angle_range=(0, 0),
        ),
    ]

    post_phase = [
        BadPhotoCopy(
            noise_type=5,
            noise_side="top",
            noise_iteration=(1, 1),
            noise_size=(1, 1),
            noise_value=(0, 15),
            noise_sparsity=(0.5, 0.5),
            noise_concentration=(0.5, 0.5),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="bottom",
            noise_iteration=(2, 3),
            noise_size=(3, 4),
            noise_value=(0, 15),
            noise_sparsity=(0.2, 0.2),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
        Geometric(rotate_range=(2, 2), randomize=0),
        Geometric(padding=[0.1, 0.2, 0.2, 0.05], randomize=0),
        BindingsAndFasteners(
            foreground=None,
            overlay_types="darken",
            effect_type="triangle_clips",
            width_range="random",
            height_range="random",
            ntimes=(5, 5),
            nscales=(0.5, 0.5),
            edge="right",
            edge_offset=(0, 0),
        ),
        BadPhotoCopy(
            noise_type=5,
            noise_side="top",
            noise_value=(0, 15),
            noise_sparsity=(0.6, 0.6),
            noise_concentration=(0.65, 0.65),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=5,
            noise_side="left",
            noise_value=(32, 64),
            noise_sparsity=(0.3, 0.4),
            noise_concentration=(0.15, 0.15),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
        Scribbles(
            scribbles_type="lines",
            scribbles_ink="pencil",
            scribbles_location="random",
            scribbles_size_range=(200, 300),
            scribbles_count_range=(2, 2),
            scribbles_thickness_range=(2, 6),
            scribbles_brightness_change=[64, 128],
            scribbles_color=(0, 0, 0),
            scribbles_lines_stroke_count_range=(1, 1),
        ),
        BindingsAndFasteners(
            foreground=None,
            overlay_types="darken",
            effect_type="triangle_clips",
            width_range="random",
            height_range="random",
            angle_range=(270, 270),
            ntimes=(3, 3),
            nscales=(1.5, 1.5),
            edge="bottom",
            edge_offset=(10, 10),
        ),
        BadPhotoCopy(
            noise_type=5,
            noise_side="right",
            noise_value=(0, 15),
            noise_sparsity=(0.3, 0.4),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_iteration=(1, 2),
            noise_size=(1, 2),
            noise_value=(0, 5),
            noise_sparsity=(0.7, 0.7),
            noise_concentration=(0.05, 0.05),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        Geometric(padding=[0.01, 0.01, 0.01, 0], randomize=0),
        ReflectedLight(
            reflected_light_smoothness=0.1,
            reflected_light_internal_radius_range=(0.1, 0.1),
            reflected_light_external_radius_range=(0.4, 0.4),
            reflected_light_minor_major_ratio_range=(0.9, 1.0),
            reflected_light_color=(255, 255, 255),
            reflected_light_internal_max_brightness_range=(0.25, 0.25),
            reflected_light_external_max_brightness_range=(0.25, 0.25),
            reflected_light_location=(0.6, 0.2),
            reflected_light_ellipse_angle_range=(0, 360),
            reflected_light_gaussian_kernel_size_range=(210, 310),
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype3():

    ink_phase = [
        InkBleed(
            intensity_range=(0.9, 0.9),
            kernel_size=(7, 7),
            severity=(0.3, 0.3),
        ),
    ]

    paper_phase = [
        BadPhotoCopy(
            noise_type=1,
            noise_side="all",
            noise_iteration=(1, 1),
            noise_size=(1, 1),
            noise_value=(110, 110),
            noise_sparsity=(0.1, 0.1),
            noise_concentration=(0.3, 0.4),
            blur_noise=1,
            blur_noise_kernel=(31, 31),
            edge_effect=0,
            wave_pattern=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="all",
            noise_iteration=(1, 1),
            noise_size=(1, 1),
            noise_value=(40, 50),
            noise_sparsity=(0.1, 0.1),
            noise_concentration=(0.3, 0.3),
            blur_noise=1,
            edge_effect=0,
            wave_pattern=0,
        ),
    ]

    post_phase = [
        BadPhotoCopy(
            noise_type=5,
            noise_side="right",
            noise_iteration=(2, 2),
            noise_size=(5, 7),
            noise_value=(100, 120),
            noise_sparsity=(0.2, 0.2),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(1, 1),
            noise_size=(1, 1),
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

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

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
            width_range="random",
            height_range="random",
            ntimes=(3, 3),
            nscales=(1.0, 1.0),
            edge="left",
            edge_offset=(10, 10),
        ),
        InkBleed(
            intensity_range=(0.5, 0.5),
            kernel_size=(3, 3),
            severity=(0.5, 0.5),
        ),
    ]

    paper_phase = [
        Geometric(
            padding=[0, 0, 0.03, 0.05],
            padding_type="fill",
            padding_value=(0, 0, 0),
            randomize=0,
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
            padding=[0, 0.05, 0, 0],
            randomize=0,
        ),
        Geometric(
            crop=(0.0, 0.02, 1.0, 0.99),
            randomize=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="all",
            noise_iteration=(1, 1),
            noise_size=(5, 7),
            noise_value=(0, 64),
            noise_sparsity=(0.9, 0.9),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=0,
        ),
        BadPhotoCopy(
            noise_type=5,
            noise_side="top_left",
            noise_iteration=(1, 1),
            noise_size=(3, 3),
            noise_value=(0, 64),
            noise_sparsity=(0.3, 0.3),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

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
            kernel_size=(3, 3),
            severity=(1.0, 1.0),
        ),
        Scribbles(
            scribbles_type="text",
            scribbles_ink="pen",
            scribbles_location=(0.8, 0.8),
            scribbles_size_range=(320, 320),
            scribbles_count_range=(1, 1),
            scribbles_thickness_range=(2, 2),
            scribbles_brightness_change=[0],
            scribbles_color=(0, 0, 0),
            scribbles_text="Dr Architg \n \n  Dr Lynn added that \n me to Dereene etan \n with you . He feel that \n we should stay away \n from the mystery that could \n help to .FTC on  \n In attend the president \n of the committee \n       Cloude ",
            scribbles_text_font="https://www.fontsquirrel.com/fonts/download/Windsong",
            scribbles_text_rotate_range=(15, 15),
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="right",
            noise_iteration=(3, 3),
            noise_size=(1, 3),
            noise_value=(0, 1),
            noise_sparsity=(0.99, 0.99),
            noise_concentration=(0.3, 0.3),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="left",
            noise_iteration=(2, 2),
            noise_size=(1, 2),
            noise_value=(0, 1),
            noise_sparsity=(0.7, 0.7),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=1,
        ),
        Geometric(randomize=0, padding=[0, 0.02, 0, 0]),
    ]

    paper_phase = [
        Geometric(
            padding=[0, 4, 0, 5],
            padding_type="fill",
            padding_value=(0, 0, 0),
            randomize=0,
        ),
    ]

    post_phase = [
        Geometric(randomize=0, rotate_range=(-2, -2)),
        Geometric(randomize=0, translation=(0.02, -0.05)),
        Geometric(
            padding=[6, 0, 0, 10],
            padding_type="fill",
            padding_value=(0, 0, 0),
            randomize=0,
        ),
        BindingsAndFasteners(
            overlay_types="min",
            effect_type="triangle_clips",
            width_range="random",
            height_range="random",
            angle_range=(270, 270),
            ntimes=(2, 2),
            nscales=(1.5, 1.5),
            edge="bottom",
            edge_offset=(20, 20),
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype6():

    ink_phase = [
        Markup(
            num_lines_range=(2, 2),
            markup_length_range=(0.4, 0.4),
            markup_thickness_range=(2, 3),
            markup_type="strikethrough",
            markup_color=(0, 0, 0),
            single_word_mode=False,
            large_word_mode=False,
        ),
    ]
    paper_phase = [
        Geometric(
            padding=[0, 0, 0, 2],
            padding_type="fill",
            padding_value=(0, 0, 0),
            randomize=0,
        ),
        Letterpress(
            n_samples=(200, 300),
            n_clusters=(500, 680),
            std_range=(500, 500),
            value_range=(245, 255),
            value_threshold_range=(128, 128),
            blur=0,
        ),
        Geometric(translation=(0, -0.05), randomize=0),
    ]
    post_phase = [
        Dithering(dither="ordered", order=(4, 4)),
        InkBleed(
            intensity_range=(0.1, 0.2),
            kernel_size=(3, 3),
            severity=(0.2, 0.2),
        ),
        Geometric(rotate_range=(-1, -1), randomize=0),
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(4, 5),
            noise_size=(1, 4),
            noise_value=(0, 0),
            noise_sparsity=(0.99, 0.99),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype7():

    ink_phase = [
        Geometric(padding=[0, 0, 0, 0.025], randomize=0),
    ]
    paper_phase = [
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(4, 5),
            noise_size=(4, 8),
            noise_value=(0, 5),
            noise_sparsity=(0.99, 0.99),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
    ]
    post_phase = [
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
        InkBleed(
            intensity_range=(0.6, 0.6),
            kernel_size=(3, 3),
            severity=(0.8, 0.8),
        ),
        BadPhotoCopy(
            noise_type=2,
            noise_side="all",
            noise_iteration=(2, 2),
            noise_size=(2, 3),
            noise_value=(0, 5),
            noise_sparsity=(0.05, 0.05),
            noise_concentration=(0.1, 0.1),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
        Geometric(rotate_range=(1, 1), randomize=0),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype8():

    ink_phase = [
        Scribbles(
            scribbles_type="text",
            scribbles_ink="pen",
            scribbles_location=(0.7, 0.01),
            scribbles_size_range=(300, 300),
            scribbles_count_range=(1, 1),
            scribbles_thickness_range=(1, 1),
            scribbles_brightness_change=[32],
            scribbles_color=(0, 0, 0),
            scribbles_text="-~ ETS  Literature",
            scribbles_text_font="random",
            scribbles_text_rotate_range=(0, 0),
        ),
        BadPhotoCopy(
            noise_type=1,
            noise_side="left",
            noise_iteration=(4, 5),
            noise_size=(2, 8),
            noise_value=(0, 64),
            noise_sparsity=(0.1, 0.1),
            noise_concentration=(0.7, 0.7),
            blur_noise=0,
            edge_effect=0,
            wave_pattern=0,
        ),
        BookBinding(
            shadow_radius_range=(30, 100),
            curve_range_right=(100, 100),
            curve_range_left=(100, 100),
            curve_ratio_right=(0.1, 0.1),
            curve_ratio_left=(0.1, 0.1),
            mirror_range=(0.15, 0.15),
            binding_align="random",
            binding_pages=(5, 10),
            curling_direction=0,
            backdrop_color=(0, 0, 0),
            enable_shadow=1,
            use_cache_images=0,
        ),
        Geometric(crop=(0.0, 0.0, 0.9, 0.85), randomize=0),
    ]

    paper_phase = [
        Geometric(translation=(-0.05, 0), randomize=0),
        Geometric(translation=(-0.9, 0), randomize=0),
    ]

    post_phase = [
        Geometric(padding=[0, 0, 0.05, 0], padding_value=0, randomize=0),
        Geometric(padding=[0, 0.1, 0, 0.1], padding_value=255, randomize=0),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

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
        NoisyLines(
            noisy_lines_direction=0,
            noisy_lines_location=[0],
            noisy_lines_number_range=(1, 1),
            noisy_lines_color=(0, 0, 0),
            noisy_lines_thickness_range=(1, 1),
            noisy_lines_random_noise_intensity_range=(0.5, 0.9),
            noisy_lines_length_interval_range=(0, 300),
            noisy_lines_gaussian_kernel_value_range=(1, 1),
            noisy_lines_overlay_method="ink_to_paper",
        ),
        Geometric(translation=(0, 0.95), randomize=0),
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(2, 3),
            noise_size=(1, 3),
            noise_value=(0, 1),
            noise_sparsity=(0.9, 0.9),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
    ]
    post_phase = [
        Geometric(rotate_range=(-1, -1), randomize=0),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype10():

    ink_phase = [
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(6, 8),
            noise_size=(1, 7),
            noise_value=(0, 1),
            noise_sparsity=(0.99, 0.99),
            noise_concentration=(0.01, 0.01),
            blur_noise=0,
            wave_pattern=0,
            edge_effect=0,
        ),
        Letterpress(
            n_samples=(200, 300),
            n_clusters=(500, 680),
            std_range=(2500, 2500),
            value_range=(245, 255),
            value_threshold_range=(128, 128),
            blur=1,
        ),
    ]
    paper_phase = []
    post_phase = [
        Faxify(
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline


def pipeline_archetype11():

    ink_phase = [
        LinesDegradation(
            line_roi=(0.0, 0.0, 1.0, 1.0),
            line_gradient_range=(32, 255),
            line_gradient_direction=(0, 2),
            line_split_probability=(0.2, 0.4),
            line_replacement_value=(5, 10),
            line_min_length=(30, 40),
            line_long_to_short_ratio=(5, 7),
            line_replacement_probability=(0.4, 0.5),
            line_replacement_thickness=(1, 3),
            p=0.5,
        ),
        InkBleed(
            intensity_range=(0.1, 0.4),
            kernel_size=(7, 7),
            severity=(0.4, 0.6),
            p=0.5,
        ),
        LowInkRandomLines(
            count_range=(3, 12),
            use_consistent_lines=False,
            noise_probability=0.16,
            p=0.5,
        ),
    ]
    paper_phase = [
        PatternGenerator(
            imgx=random.randint(256, 512),
            imgy=random.randint(256, 512),
            n_rotation_range=(5, 15),
            p=0.5,
        ),
        ColorPaper(
            hue_range=(0, 255),
            saturation_range=(10, 40),
            p=0.5,
        ),
        NoiseTexturize(
            sigma_range=(3, 10),
            turbulence_range=(2, 5),
            p=0.5,
        ),
        BrightnessTexturize(
            texturize_range=(0.9, 0.99),
            deviation=0.03,
            p=0.5,
        ),
    ]

    post_phase = [
        ColorShift(
            color_shift_offset_x_range=(1, 2),
            color_shift_offset_y_range=(1, 2),
            color_shift_iterations=(1, 2),
            color_shift_brightness_range=(0.9, 1.1),
            color_shift_gaussian_kernel_range=(1, 1),
            p=0.5,
        ),
        SubtleNoise(
            subtle_range=10,
            p=0.5,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.5,
        ),
        ShadowCast(
            shadow_side="random",
            shadow_vertices_range=(1, 20),
            shadow_width_range=(0.3, 0.8),
            shadow_height_range=(0.3, 0.8),
            shadow_color=(64, 64, 64),
            shadow_opacity_range=(0.2, 0.9),
            shadow_iterations_range=(1, 2),
            shadow_blur_kernel_range=(101, 301),
            p=1,
        ),
        InkMottling(
            ink_mottling_alpha_range=(0.1, 0.2),
            ink_mottling_noise_scale_range=(1, 2),
            ink_mottling_gaussian_kernel_range=(3, 5),
            p=1,
        ),
        BleedThrough(
            intensity_range=(0.1, 0.2),
            color_range=(32, 224),
            ksize=(17, 17),
            sigmaX=0,
            alpha=random.uniform(0.05, 0.1),
            offsets=(10, 20),
            p=0.5,
        ),
    ]

    pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

    return pipeline

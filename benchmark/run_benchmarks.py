# import libraries
import argparse
import random
import tracemalloc
from glob import glob
from time import time

import cv2
import numpy as np
from pytablewriter import MarkdownTableWriter
from tqdm import tqdm

from augraphy import *

# set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
cv2.setRNGSeed(random_seed)


# function to run benchmarks
def run_benchmarks(augmentations, folder_path):

    # check jpg, png and bmp for now
    file_paths = []
    for file_type in ["jpg", "png", "bmp"]:
        file_paths += glob(folder_path + "/*." + file_type, recursive=True)

    augmentations_name = []
    augmentations_augment_time = []
    augmentations_memory_usage = []
    augmentations_augment_time_mask = []
    augmentations_memory_usage_mask = []
    augmentations_augment_time_keypoints = []
    augmentations_memory_usage_keypoints = []
    augmentations_augment_time_bounding_boxes = []
    augmentations_memory_usage_bounding_boxes = []

    for augmentation in augmentations:

        print("Running augmentation for " + augmentation.__name__)

        # image
        augmentation_times = []
        augmentations_memory = []
        # mask
        augmentation_times_mask = []
        augmentations_memory_mask = []
        # keypoints
        augmentation_times_keypoints = []
        augmentations_memory_keypoints = []
        # bounding boxes
        augmentation_times_bounding_boxes = []
        augmentations_memory_bounding_boxes = []

        for i in tqdm(range(len(file_paths))):
            image = cv2.imread(file_paths[i])

            if len(image.shape) > 2:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image
            otsu_threshold, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # create mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
            mask[image_binary > 0] = 255

            # create 1000 random keypoints, under 5 different names
            keypoints = {}
            for points_name in ("points1", "points2", "points3", "points4", "points5"):
                n = 0
                points = []
                while n < 200:
                    points.append([random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)])
                    n += 1
                keypoints[points_name] = points

            # create 100 random size bounding boxes
            bounding_boxes = []
            for i in range(100):
                width = random.randint(10, image.shape[1] - 10)
                height = random.randint(10, image.shape[0] - 10)
                xs = random.randint(0, image.shape[1] - width)
                ys = random.randint(0, image.shape[0] - height)
                xe = xs + width
                ye = ys + height
                bounding_boxes.append([xs, ys, xe, ye])

            # image
            tracemalloc.start()
            memory_list_start = tracemalloc.get_traced_memory()
            start_time = time()
            # use default params
            augmentor = augmentation(p=1)
            # apply augmentation
            output = augmentor(image)
            memory_list_stop = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # memory consumed = maximum memory - initial memory
            augmentations_memory += [memory_list_stop[1] - memory_list_start[0]]
            # time to augment the image
            augmentation_times += [time() - start_time]

            # mask
            tracemalloc.start()
            memory_list_start = tracemalloc.get_traced_memory()
            start_time = time()
            # use default params
            augmentor = augmentation(p=1)
            # apply augmentation
            output = augmentor(image, mask=mask)
            memory_list_stop = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # memory consumed = maximum memory - initial memory
            augmentations_memory_mask += [memory_list_stop[1] - memory_list_start[0]]
            # time to augment the image
            augmentation_times_mask += [time() - start_time]

            # keypoints
            tracemalloc.start()
            memory_list_start = tracemalloc.get_traced_memory()
            start_time = time()
            # use default params
            augmentor = augmentation(p=1)
            # apply augmentation
            output = augmentor(image, keypoints=keypoints)
            memory_list_stop = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # memory consumed = maximum memory - initial memory
            augmentations_memory_keypoints += [memory_list_stop[1] - memory_list_start[0]]
            # time to augment the image
            augmentation_times_keypoints += [time() - start_time]

            # bounding boxes
            tracemalloc.start()
            memory_list_start = tracemalloc.get_traced_memory()
            start_time = time()
            # use default params
            augmentor = augmentation(p=1)
            # apply augmentation
            output = augmentor(image, bounding_boxes=bounding_boxes)
            memory_list_stop = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # memory consumed = maximum memory - initial memory
            augmentations_memory_bounding_boxes += [memory_list_stop[1] - memory_list_start[0]]
            # time to augment the image
            augmentation_times_bounding_boxes += [time() - start_time]

        # image
        average_augmentation_time = np.mean(augmentation_times)
        average_memory_usage = np.mean(augmentations_memory)

        # mask
        average_augmentation_time_mask = np.mean(augmentation_times_mask)
        average_memory_usage_mask = np.mean(augmentations_memory_mask)

        # keypoints
        average_augmentation_time_keypoints = np.mean(augmentation_times_keypoints)
        average_memory_usage_keypoints = np.mean(augmentations_memory_keypoints)

        # bounding boxes
        average_augmentation_time_bounding_boxes = np.mean(augmentation_times_bounding_boxes)
        average_memory_usage_bounding_boxes = np.mean(augmentations_memory_bounding_boxes)

        augmentations_name += [augmentation.__name__]
        augmentations_augment_time += [average_augmentation_time]
        augmentations_memory_usage += [average_memory_usage]
        augmentations_augment_time_mask += [average_augmentation_time_mask]
        augmentations_memory_usage_mask += [average_memory_usage_mask]
        augmentations_augment_time_keypoints += [average_augmentation_time_keypoints]
        augmentations_memory_usage_keypoints += [average_memory_usage_keypoints]
        augmentations_augment_time_bounding_boxes += [average_augmentation_time_bounding_boxes]
        augmentations_memory_usage_bounding_boxes += [average_memory_usage_bounding_boxes]

        print("Average time per augmentation = " + str(average_augmentation_time) + " seconds")
        print("Average memory usage per augmentation = " + str(average_memory_usage / 1e6) + " MB")
        print("Average time per augmentation (mask) = " + str(average_augmentation_time_mask) + " seconds")
        print("Average memory usage per augmentation (mask) = " + str(average_memory_usage_mask / 1e6) + " MB")
        print("Average time per augmentation (keypoints) = " + str(average_augmentation_time_keypoints) + " seconds")
        print(
            "Average memory usage per augmentation (keypoints) = " + str(average_memory_usage_keypoints / 1e6) + " MB",
        )
        print(
            "Average time per augmentation (bounding boxes) = "
            + str(average_augmentation_time_bounding_boxes)
            + " seconds",
        )
        print(
            "Average memory usage per augmentation (bounding boxes) = "
            + str(average_memory_usage_bounding_boxes / 1e6)
            + " MB",
        )
        print(" ")

    return (
        augmentations_name,
        augmentations_augment_time,
        augmentations_memory_usage,
        augmentations_augment_time_mask,
        augmentations_memory_usage_mask,
        augmentations_augment_time_keypoints,
        augmentations_memory_usage_keypoints,
        augmentations_augment_time_bounding_boxes,
        augmentations_memory_usage_bounding_boxes,
    )


def generate_markdown_table(augmentations_name, augmentations_data):

    (
        augmentations_augment_time,
        augmentations_memory_usage,
        augmentations_augment_time_mask,
        augmentations_memory_usage_mask,
        augmentations_augment_time_keypoints,
        augmentations_memory_usage_keypoints,
        augmentations_augment_time_bounding_boxes,
        augmentations_memory_usage_bounding_boxes,
    ) = augmentations_data

    table_name = "Benchmarking results"
    headers = [
        "Augmentation",
        "Img/sec",
        "Memory(MB)",
        "Img/sec(mask)",
        "Memory(MB)(mask)",
        "Img/sec(keypoints)",
        "Memory(MB)(keypoints)",
        "Img/sec(bounding boxes)",
        "Memory(MB)(bounding boxes)",
    ]
    value_matrix = []
    for (
        name,
        augmentation_time,
        memory_usage,
        augmentation_time_mask,
        memory_usage_mask,
        augmentation_time_keypoints,
        memory_usage_keypoints,
        augmentation_time_bounding_boxes,
        memory_usage_bounding_boxes,
    ) in zip(
        augmentations_name,
        augmentations_augment_time,
        augmentations_memory_usage,
        augmentations_augment_time_mask,
        augmentations_memory_usage_mask,
        augmentations_augment_time_keypoints,
        augmentations_memory_usage_keypoints,
        augmentations_augment_time_bounding_boxes,
        augmentations_memory_usage_bounding_boxes,
    ):
        value_matrix.append(
            [
                name,
                np.round(1 / augmentation_time, 2),
                np.round(memory_usage / 1e6, 2),
                np.round(1 / augmentation_time_mask, 2),
                np.round(memory_usage_mask / 1e6, 2),
                np.round(1 / augmentation_time_keypoints, 2),
                np.round(memory_usage_keypoints / 1e6, 2),
                np.round(1 / augmentation_time_bounding_boxes, 2),
                np.round(memory_usage_bounding_boxes / 1e6, 2),
            ],
        )

    writer = MarkdownTableWriter(
        table_name=table_name,
        headers=headers,
        value_matrix=value_matrix,
    )

    writer.write_table()


def main(folder_path):

    # augmentations to be applied
    augmentations = [
        BadPhotoCopy,
        BindingsAndFasteners,
        BleedThrough,
        BookBinding,
        Brightness,
        BrightnessTexturize,
        ColorPaper,
        ColorShift,
        DelaunayTessellation,
        DepthSimulatedBlur,
        DirtyDrum,
        DirtyRollers,
        DirtyScreen,
        Dithering,
        DotMatrix,
        DoubleExposure,
        Faxify,
        Folding,
        Gamma,
        Geometric,
        GlitchEffect,
        Hollow,
        InkBleed,
        InkColorSwap,
        InkMottling,
        InkShifter,
        LCDScreenPattern,
        Jpeg,
        LensFlare,
        Letterpress,
        LightingGradient,
        LinesDegradation,
        LowInkPeriodicLines,
        LowInkRandomLines,
        LowLightNoise,
        Markup,
        Moire,
        NoiseTexturize,
        NoisyLines,
        PageBorder,
        PatternGenerator,
        ReflectedLight,
        Scribbles,
        SectionShift,
        ShadowCast,
        Squish,
        Stains,
        SubtleNoise,
        VoronoiTessellation,
        WaterMark,
    ]

    # run and process image with each augmentation
    (
        augmentations_name,
        augmentations_augment_time,
        augmentations_memory_usage,
        augmentations_augment_time_mask,
        augmentations_memory_usage_mask,
        augmentations_augment_time_keypoints,
        augmentations_memory_usage_keypoints,
        augmentations_augment_time_bounding_boxes,
        augmentations_memory_usage_bounding_boxes,
    ) = run_benchmarks(augmentations, folder_path)

    # generate table
    generate_markdown_table(
        augmentations_name,
        [
            augmentations_augment_time,
            augmentations_memory_usage,
            augmentations_augment_time_mask,
            augmentations_memory_usage_mask,
            augmentations_augment_time_keypoints,
            augmentations_memory_usage_keypoints,
            augmentations_augment_time_bounding_boxes,
            augmentations_memory_usage_bounding_boxes,
        ],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the folder of images")
    opt = parser.parse_args()

    main(opt.folder_path)

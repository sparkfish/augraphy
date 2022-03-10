import os
import random
import time
from glob import glob

import cv2
import numpy as np

from augraphy.augmentations.lib import make_white_transparent
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.base.augmentationsequence import AugmentationSequence


class AugraphyPipeline:
    """Contains phases of image augmentations and their results.

    :param ink_phase: Collection of Augmentations to apply.
    :type ink_phase: base.augmentationsequence or list
    :param paper_phase: Collection of Augmentations to apply.
    :type paper_phase: base.augmentationsequence or list
    :param post_phase: Collection of Augmentations to apply.
    :type post_phase: base.augmentationsequence or list
    :param ink_color_range: Pair of ints determining the range from which to
           sample the ink color.
    :type ink_color_range: tuple, optional
    :param paper_color_range: Pair of ints determining the range from which to
           sample the paper color.
    :type paper_color_range: tuple, optional
    :param log_prob: Flag to enable logging of each augmentation probablity.
    :type log_prob: bool, optional
    """

    def __init__(
        self,
        ink_phase,
        paper_phase,
        post_phase,
        ink_color_range=(0, 0),
        paper_color_range=(255, 255),
        log_prob=False,
    ):
        """Constructor method"""
        self.ink_phase = self.wrapListMaybe(ink_phase)
        self.paper_phase = self.wrapListMaybe(paper_phase)
        self.post_phase = self.wrapListMaybe(post_phase)
        self.ink_color_range = ink_color_range
        self.paper_color_range = paper_color_range
        self.log_prob = log_prob

        # create directory to store log files
        if self.log_prob:
            self.log_prob_path = os.path.join(os.getcwd(), "log_prob/")
            os.makedirs(self.log_prob_path, exist_ok=True)

    def wrapListMaybe(self, augs):
        """Converts a bare list to an AugmentationSequence, or does nothing."""
        if type(augs) is list:
            return AugmentationSequence(augs)
        else:
            return augs

    def augment(self, image):
        """Applies the Augmentations in each phase of the pipeline.

        :param image: The image to apply Augmentations to. Minimum 30x30 pixels.
        :type image: numpy.array
        :return: A dictionary of AugmentationResults representing the changes
                 in each phase of the pipeline.
        :rtype: dictionary
        """

        # Check that image is the correct size.
        if (image.shape[0] < 30) or (image.shape[1] < 30):
            raise Exception(
                "Image should have dimensions greater than 30x30, but actual dimensions were {}.".format(
                    image.shape,
                ),
            )

        # create augraphy cache folder
        cache_folder_path = os.path.join(os.getcwd() + "/augraphy_cache/")
        os.makedirs(cache_folder_path, exist_ok=True)
        cache_image_paths = glob(cache_folder_path + "*.png", recursive=True)

        file_indices = []
        modified_time = []
        for image_path in cache_image_paths:
            file_name = os.path.basename(image_path)
            file_indices.append(int(file_name[file_name.index("_") + 1 : -4]))
            modified_time.append(os.path.getmtime(image_path))

        # store 30 cache image files
        if len(cache_image_paths) >= 30:
            oldest_index = np.argmin(modified_time)
            cv2.imwrite(cache_folder_path + "image_" + str(file_indices[oldest_index]) + ".png", image)
        else:
            current_image_index = len(cache_image_paths)
            cv2.imwrite(cache_folder_path + "image_" + str(current_image_index) + ".png", image)

        data = dict()

        # Store performance metadata and other logs here.
        data["log"] = dict()

        # For storing augmentation execution times.
        data["log"]["time"] = list()
        data["log"]["augmentation_name"] = list()
        data["log"]["augmentation_status"] = list()

        # This is useful.
        data["log"]["image_shape"] = image.shape

        data["image"] = image.copy()
        ink = data["image"].copy()

        data["pipeline"] = self
        data["ink"] = list()
        data["paper"] = list()
        data["post"] = list()

        data["ink"].append(AugmentationResult(None, ink))

        if (self.paper_color_range[0] != 0) | (self.paper_color_range[1] != 0):
            paper_color = random.randint(
                self.paper_color_range[0],
                self.paper_color_range[1],
            )
        else:
            paper_color = 255

        data["log"]["paper_color"] = paper_color

        data["paper"].append(
            AugmentationResult(
                None,
                np.full(
                    (ink.shape[0], ink.shape[1], 3),
                    paper_color,
                    dtype=np.uint8,
                ),
            ),
        )

        # If phases were defined None or [] in a custom pipeline, they wouldn't
        # be callable objects, so make them empty AugmentationSequences
        if self.ink_phase is None or self.ink_phase == []:
            self.ink_phase = AugmentationSequence([])

        if self.paper_phase is None or self.paper_phase == []:
            self.paper_phase = AugmentationSequence([])

        if self.post_phase is None or self.post_phase == []:
            self.post_phase = AugmentationSequence([])

        # apply ink phase augmentation
        self.apply_phase(data, layer="ink", phase=self.ink_phase)

        # apply paper phase augmentations
        self.apply_phase(data, layer="paper", phase=self.paper_phase)

        # ink and paper phases always have at least one result by now
        data["post"].append(
            AugmentationResult(
                None,
                self.print_ink_to_paper(
                    data,
                    data["ink"][-1].result,
                    data["paper"][-1].result,
                ),
            ),
        )

        # apply post phase augmentations
        self.apply_phase(data, layer="post", phase=self.post_phase)

        data["output"] = data["post"][-1].result.astype("uint8")

        # log probability
        if self.log_prob:

            # path to log file
            log_file_name = "log_prob_" + time.strftime("%H:%M:%S", time.localtime()) + ".txt"
            log_file_name = log_file_name.replace(":", "_")
            log_prob_file_path = self.log_prob_path + log_file_name

            augmentation_names = data["log"]["augmentation_name"]
            augmentation_status = data["log"]["augmentation_status"]
            with open(log_prob_file_path, "w+") as file:
                for (
                    name,
                    status,
                ) in zip(augmentation_names, augmentation_status):
                    file.write("%s,%s \n" % (name, status))
            file.close()

        return data

    def apply_phase(self, data, layer, phase):
        """Applies every augmentation in a phase."""
        for augmentation in phase.augmentations:
            result = data[layer][-1].result.copy()

            if augmentation.should_run():
                start = time.process_time()  # time at start of execution
                result = augmentation(result, layer, force=True)
                end = time.process_time()  # time at end of execution
                elapsed = end - start  # execution duration
                data["log"]["time"].append((augmentation, elapsed))
            else:
                result = None

            data["log"]["augmentation_name"].append(augmentation.__class__.__name__)
            if result is None:
                data["log"]["augmentation_status"].append(False)
                data[layer].append(
                    AugmentationResult(
                        augmentation,
                        data[layer][-1].result.copy(),
                        'This augmentation did not run, its "result" is unchanged.',
                    ),
                )
            else:
                data["log"]["augmentation_status"].append(True)
                # for "OneOf" or "AugmentationSequence"
                while isinstance(result, tuple) or isinstance(result, list):
                    result, augmentations = result
                    for nested_augmentation in augmentations:
                        data["log"]["augmentation_name"].append(nested_augmentation.__class__.__name__)
                        data["log"]["augmentation_status"].append(True)
                data[layer].append(AugmentationResult(augmentation, result))

    def print_ink_to_paper(self, data, overlay, background):
        """Applies the ink layer to the paper layer."""

        # prevent inconsistency in size between background and overlay
        if overlay.shape[:2] != background.shape[:2]:
            overlay_y, overlay_x = overlay.shape[:2]
            background = cv2.resize(background, (overlay_x, overlay_y), interpolation=cv2.INTER_AREA)

        if (self.ink_color_range[0] != 0) or (self.ink_color_range[1] != 0):
            ink_color = random.randint(self.ink_color_range[0], self.ink_color_range[1])
        else:
            ink_color = 0

        data["log"]["ink_color"] = ink_color

        overlay = make_white_transparent(overlay, ink_color)
        # Split out the transparency mask from the colour info
        overlay_img = overlay[:, :, :3]  # Grab the BRG planes
        overlay_mask = overlay[:, :, 3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the single channel alpha masks into three channel, so we can use them as weights
        if len(overlay.shape) > 2:
            overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        if len(background_mask.shape) > 2:
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        # Convert background to 3 channels if they are in single channel
        if len(background.shape) < 3:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        background_part = (background * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(
            cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0),
        )

    def __repr__(self):
        r = f"ink_phase = {repr(self.ink_phase)}\n\n"
        r += f"paper_phase = {repr(self.ink_phase)}\n\n"
        r += f"post_phase = {repr(self.ink_phase)}\n\n"

        r += f"AugraphyPipeline(ink_phase, paper_phase, post_phase, ink_color_range={self.ink_color_range}, paper_color_range={self.paper_color_range})"

        return r

    def visualize(self):
        print(repr(self))

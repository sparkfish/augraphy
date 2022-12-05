import os
import random
import time
from copy import copy
from copy import deepcopy
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
    :param save_outputs: Flag to enable saving each phase output image.
    :type save_outputs: bool, optional
    :param log: Flag to enable logging.
    :type log: bool, optional
    :param random_seed: The initial value for PRNGs used in Augraphy.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        ink_phase,
        paper_phase,
        post_phase,
        ink_color_range=(-1, -1),
        paper_color_range=(255, 255),
        save_outputs=False,
        log=False,
        random_seed=None,
    ):
        """Constructor method"""
        self.ink_phase = self.wrapListMaybe(ink_phase)
        self.paper_phase = self.wrapListMaybe(paper_phase)
        self.post_phase = self.wrapListMaybe(post_phase)
        self.ink_color_range = ink_color_range
        self.paper_color_range = paper_color_range
        self.save_outputs = save_outputs
        self.log = log
        self.random_seed = random_seed

        # ensure determinism if random_seed set
        if self.random_seed:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            cv2.setRNGSeed(self.random_seed)

        # create directory to store log files
        if self.log:
            self.log_prob_path = os.path.join(os.getcwd(), "logs/")
            os.makedirs(self.log_prob_path, exist_ok=True)

        if self.save_outputs:
            # create each phase folder
            self.save_paths = []
            self.save_paths.append(os.path.join(os.getcwd(), "augmentation_images/ink/"))
            self.save_paths.append(os.path.join(os.getcwd(), "augmentation_images/paper/"))
            self.save_paths.append(os.path.join(os.getcwd(), "augmentation_images/post/"))
            for i in range(len(self.save_paths)):
                os.makedirs(self.save_paths[i], exist_ok=True)

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

        # Check if image has correct channel
        if len(image.shape) > 2 and (image.shape[2] != 3):
            raise Exception(
                "Image should have channel number of 3 (BGR), but actual dimensions were {}.".format(
                    image.shape,
                ),
            )
        # Check that image is the correct size.
        elif (image.shape[0] < 30) or (image.shape[1] < 30):
            raise Exception(
                "Image should have dimensions greater than 30x30, but actual dimensions were {}.".format(
                    image.shape,
                ),
            )

        # get and check valid image type ( uint or float)
        image_type = str(image.dtype)
        image_max_value = 255
        if image_type[:5] == "float":
            if np.max(image) <= 1:
                image_max_value = 1
                image = np.uint8(image * 255)
            else:
                image = np.uint8(image)
        elif image_type[:4] != "uint":
            raise Exception(
                "Image type should be uint or float, but the image type is {}.".format(
                    image_type,
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
            cv2.imwrite(
                cache_folder_path + "image_" + str(file_indices[oldest_index]) + ".png",
                image,
            )
        else:
            current_image_index = len(cache_image_paths)
            cv2.imwrite(
                cache_folder_path + "image_" + str(current_image_index) + ".png",
                image,
            )

        data = dict()

        # Store performance metadata and other logs here.
        data["log"] = dict()

        # For storing augmentation execution times.
        data["log"]["time"] = list()
        data["log"]["augmentation_name"] = list()
        data["log"]["augmentation_status"] = list()
        data["log"]["augmentation_parameters"] = list()

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

        # revert to input image type
        if image_type[:5] == "float":
            if image_max_value == 1:
                data["output"] = (data["post"][-1].result.astype(image_type)) / 255
            else:
                data["output"] = data["post"][-1].result.astype(image_type)
        else:
            data["output"] = data["post"][-1].result.astype("uint8")

        # save each phase augmented images
        if self.save_outputs:
            self.save_images(data)

        # log probability
        if self.log:
            self.write_log(data)

        return data

    def save_images(self, data):
        """Save each augmented image in each phases to local disk.

        :param data: A dictionary of AugmentationResults representing the changes in each phase of the pipeline.
        :type data: dictionary
        """

        layer_names = ["ink", "paper", "post"]
        ink_layers = data["ink"]
        paper_layers = data["paper"]
        post_layers = data["post"]

        n = 0
        for j, layers in enumerate([ink_layers, paper_layers, post_layers]):

            # output path for each ink, paper and post phase images
            save_path = self.save_paths[j]

            # name of layer or phase
            layer_name = layer_names[j]

            for i, layer_data in enumerate(layers):
                if layer_data.metadata is None:
                    result = layer_data.result

                    # input layer
                    if layer_data.augmentation is None:
                        augmentation_name = layer_name + "_layer_input"
                        cv2.imwrite(
                            save_path + "p" + str(n) + "_" + layer_name + str(i) + "_" + augmentation_name + ".png",
                            result,
                        )
                        n += 1

                    # one of
                    elif layer_data.augmentation.__class__.__name__ == "OneOf":
                        augmentation_name = "oneof_"
                        n = self.get_oneof_data(
                            layer_data.augmentation,
                            result,
                            save_path,
                            layer_name,
                            augmentation_name,
                            i,
                            n,
                        )

                    # sequence
                    elif layer_data.augmentation.__class__.__name__ == "AugmentationSequence":
                        augmentation_name = "sequence_"
                        n = self.get_sequence_data(
                            layer_data.augmentation,
                            result,
                            save_path,
                            layer_name,
                            augmentation_name,
                            i,
                            n,
                        )

                    # normal augmentations
                    else:
                        augmentation_name = layer_data.augmentation.__class__.__name__
                        cv2.imwrite(
                            save_path + "p" + str(n) + "_" + layer_name + str(i) + "_" + augmentation_name + ".png",
                            result,
                        )
                        n += 1

    def get_oneof_data(self, augmentation, result, save_path, layer_name, augmentation_name, i, n):
        """Get augmentation information from OneOf augmentation recursively or save the augmented image in disk.

        :param augmentation: Augmentation object of OneOf augmentation.
        :type augmentation: class instance
        :param result: Augmentation output, it may be nested in a list.
        :type result: list or numpy array.
        :param save_path: Output path of result.
        :type save_path: list
        :param layer_name: Name of current layer.
        :type layer_name: list
        :param augmentation_name: A combined name of augmentations, seperated by _.
        :type augmentation_name: list
        :param i: Index of current augmentation in total number of augmentations.
        :type i: int
        :param n: Index of current augmented in total number of augmented images.
        :type n: int
        """

        current_augmentation = augmentation.augmentations[np.argmax(augmentation.augmentation_probabilities)]
        # sequence inside oneof
        if current_augmentation.__class__.__name__ == "AugmentationSequence":
            augmentation_name += "sequence_"
            n = self.get_sequence_data(current_augmentation, result, save_path, layer_name, augmentation_name, i, n)
        # oneof inside oneof
        elif current_augmentation.__class__.__name__ == "OneOf":
            augmentation_name += "oneof_"
            n = self.get_oneof_data(current_augmentation, result, save_path, layer_name, augmentation_name, i, n)
        # augmentations inside oneof
        else:
            augmentation_name += current_augmentation.__class__.__name__
            cv2.imwrite(save_path + "p" + str(n) + "_" + layer_name + str(i) + "_" + augmentation_name + ".png", result)
            n += 1
        return n

    def get_sequence_data(self, augmentation, result, save_path, layer_name, input_augmentation_name, i, n):
        """Get augmentation information from AugmentationSequence augmentation recursively or save the augmented image in disk.

        :param augmentation: Augmentation object of OneOf augmentation.
        :type augmentation: class instance
        :param result: Augmentation output, it may be nested in a list.
        :type result: list or numpy array.
        :param save_path: Output path of result.
        :type save_path: list
        :param layer_name: Name of current layer.
        :type layer_name: list
        :param augmentation_name: A combined name of augmentations, seperated by _.
        :type augmentation_name: list
        :param i: Index of current augmentation in total number of augmentations.
        :type i: int
        :param n: Index of current augmented in total number of augmented images.
        :type n: int
        """

        s = 0
        for current_augmentation, result in zip(augmentation.augmentations, augmentation.results):
            augmentation_name = copy(input_augmentation_name) + str(s) + "_"
            # sequence inside sequence
            if current_augmentation.__class__.__name__ == "AugmentationSequence":
                # sequence returns (result, self.augmentations), so get result only here
                result = result[0]
                augmentation_name += "sequence_"
                n = self.get_sequence_data(current_augmentation, result, save_path, layer_name, augmentation_name, i, n)
            # oneof inside sequence
            elif current_augmentation.__class__.__name__ == "OneOf":
                # oneof returns (image, [augmentation]), so get image only here
                result = result[0]
                augmentation_name += "oneof_"
                n = self.get_oneof_data(current_augmentation, result, save_path, layer_name, augmentation_name, i, n)
            # augmentations inside sequence
            else:
                augmentation_name += current_augmentation.__class__.__name__
                cv2.imwrite(
                    save_path + "p" + str(n) + "_" + layer_name + str(i) + "_" + augmentation_name + ".png",
                    result,
                )
                n += 1
            s += 1
        return n

    def write_log(self, data):
        """Save augmentations log to local disk.

        :param data: A dictionary of AugmentationResults representing the changes in each phase of the pipeline.
        :type data: dictionary
        """

        # path to log file
        log_file_name = "log_" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".txt"
        log_prob_file_path = self.log_prob_path + log_file_name

        augmentation_names = data["log"]["augmentation_name"]
        augmentation_status = data["log"]["augmentation_status"]
        augmentation_parameters = deepcopy(data["log"]["augmentation_parameters"])

        # remove image array and replace it with shape
        for j, augmentation_parameter in enumerate(augmentation_parameters):
            # check and convert from tuple to list
            if isinstance(augmentation_parameter, tuple):
                augmentation_parameter = list(augmentation_parameter)
                augmentation_parameters[j] = augmentation_parameter
            check_values = [augmentation_parameter]
            while check_values:
                value = check_values.pop(0)
                if value:
                    if isinstance(value, list):
                        for i, nested_value in enumerate(value):
                            if hasattr(nested_value, "shape"):
                                value[i] = nested_value.shape
                            elif (
                                isinstance(nested_value, list)
                                or isinstance(nested_value, tuple)
                                or hasattr(nested_value, "shape")
                            ):
                                # convert from tuple to list
                                if isinstance(nested_value, tuple):
                                    nested_value = list(nested_value)
                                    value[i] = nested_value
                                check_values.append(nested_value)
                    elif hasattr(value, "items"):
                        for parameter, nested_value in value.items():
                            if hasattr(nested_value, "shape"):
                                value[parameter] = nested_value.shape
                            elif (
                                isinstance(nested_value, list)
                                or isinstance(nested_value, tuple)
                                or hasattr(nested_value, "shape")
                            ):
                                # convert from tuple to list
                                if isinstance(nested_value, tuple):
                                    nested_value = list(nested_value)
                                    value[parameter] = nested_value
                                check_values.append(nested_value)

        with open(log_prob_file_path, "w+") as file:
            for (name, status, parameters) in zip(
                augmentation_names,
                augmentation_status,
                augmentation_parameters,
            ):
                file.write("%s,%s,%s \n" % (name, status, parameters))
                # put a space
                file.write("\n")
        file.close()

    def apply_phase(self, data, layer, phase):
        """Applies every augmentation in a phase.

        :param data: A dictionary of AugmentationResults representing the changes in each phase of the pipeline.
        :type data: dictionary
        :param layer: The name of current layer or phase.
        :type layer: string
        :param phase: Collection of Augmentations to apply.
        :type phase: base.augmentationsequence or list
        """
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
                data["log"]["augmentation_parameters"].append("")
                data[layer].append(
                    AugmentationResult(
                        augmentation,
                        data[layer][-1].result.copy(),
                        'This augmentation did not run, its "result" is unchanged.',
                    ),
                )
            else:
                data["log"]["augmentation_status"].append(True)
                data["log"]["augmentation_parameters"].append(augmentation.__dict__)
                # for "OneOf" or "AugmentationSequence"
                while isinstance(result, tuple) or isinstance(result, list):
                    result, augmentations = result
                    for nested_augmentation in augmentations:
                        data["log"]["augmentation_name"].append(
                            nested_augmentation.__class__.__name__,
                        )
                        data["log"]["augmentation_status"].append(True)
                        data["log"]["augmentation_parameters"].append(
                            nested_augmentation.__dict__,
                        )
                data[layer].append(AugmentationResult(augmentation, result))

    def print_ink_to_paper(self, data, overlay, background):
        """Applies the ink layer to the paper layer.

        :param data: A dictionary of AugmentationResults representing the changes in each phase of the pipeline.
        :type data: dictionary
        :param overlay: Foreground of overlay process, output from ink phase.
        :type overlay: numpy array
        :param background: Background of overlay process, output from paper phase.
        :type background: numpy array
        """
        # prevent inconsistency in size between background and overlay
        if overlay.shape[:2] != background.shape[:2]:
            overlay_y, overlay_x = overlay.shape[:2]
            background = cv2.resize(
                background,
                (overlay_x, overlay_y),
                interpolation=cv2.INTER_AREA,
            )

        if (self.ink_color_range[0] != -1) or (self.ink_color_range[1] != -1):
            ink_color = random.randint(self.ink_color_range[0], self.ink_color_range[1])
        else:
            ink_color = -1

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
        background_part = (background * (1.0 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1.0 / 255.0)) * (overlay_mask * (1.0 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(
            cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0),
        )

    def __repr__(self):
        r = f"ink_phase = {repr(self.ink_phase)}\n\n"
        r += f"paper_phase = {repr(self.paper_phase)}\n\n"
        r += f"post_phase = {repr(self.post_phase)}\n\n"

        r += f"AugraphyPipeline(ink_phase, paper_phase, post_phase, ink_color_range={self.ink_color_range}, paper_color_range={self.paper_color_range})"

        return r

    def visualize(self):
        print(repr(self))

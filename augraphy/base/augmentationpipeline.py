import random

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
    :param rotate_range: Pair of ints determining the range from which to sample
           the paper rotation.
    :type rotate_range: tuple, optional
    """

    def __init__(
        self,
        ink_phase,
        paper_phase,
        post_phase,
        ink_color_range=(0, 0),
        paper_color_range=(255, 255),
        rotate_range=(0, 0),
    ):
        """Constructor method"""
        self.ink_phase = self.wrapListMaybe(ink_phase)
        self.paper_phase = self.wrapListMaybe(paper_phase)
        self.post_phase = self.wrapListMaybe(post_phase)
        self.ink_color_range = ink_color_range
        self.rotate_range = rotate_range
        self.paper_color_range = paper_color_range

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

        data = dict()

        # Store performance metadata and other logs here.
        data["log"] = dict()

        # For storing augmentation execution times.
        data["log"]["time"] = list()

        # This is useful.
        data["log"]["image_shape"] = image.shape

        data["image"] = image.copy()
        ink = data["image"].copy()

        if (self.rotate_range[0] != 0) | (self.rotate_range[1] != 0):
            angle = random.randint(self.rotate_range[0], self.rotate_range[1])
        else:
            angle = 0

        data["log"]["rotation_angle"] = angle

        if angle != 0:
            ink = self.rotate_image(ink, angle)
        else:
            ink = ink

        data["image_rotated"] = ink.copy()

        if len(ink.shape) > 2 and ink.shape[2] == 3:
            ink = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
        elif len(ink.shape) > 2 and ink.shape[2] == 4:
            ink = cv2.cvtColor(ink, cv2.COLOR_BGRA2GRAY)

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
                    dtype="uint",
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

        self.ink_phase(data)
        self.paper_phase(data)

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

        self.post_phase(data)

        data["output"] = data["post"][-1].result.astype("uint8")

        return data

    def rotate_image(self, mat, angle):
        """Rotates an image (angle in degrees) and expands image to avoid
        cropping.
        """
        mat = cv2.bitwise_not(mat)
        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2,
        )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        rotated_mat = cv2.bitwise_not(rotated_mat)
        return rotated_mat

    def print_ink_to_paper(self, data, overlay, background):
        """Applies the ink layer to the paper layer."""

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

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

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

        r += f"AugraphyPipeline(ink_phase, paper_phase, post_phase, ink_color_range={self.ink_color_range}, paper_color_range={self.paper_color_range}, rotate_range={self.rotate_range})"

        return r

    def visualize(self):
        print(repr(self))

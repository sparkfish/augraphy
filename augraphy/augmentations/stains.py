import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.overlaybuilder import OverlayBuilder
from augraphy.utilities.texturegenerator import TextureGenerator


class Stains(Augmentation):
    """Creates a stains texture and the stains texture is blended into the input image by using OverlayBuilder.


    :param stains_type: Types of stains. Use "random" for random stains effect.
        Select from "rough_stains", "fine_stains", "severe_stains", "light_stains".
    :type stains_type: tuple, optional
    :param stains_blend_method: The blending method to blend stains texture into the input image.
    :type stains_blend_method: string, optional
    :param stains_blend_alpha: The blending alpha value for blending method with the usage of alpha.
    :type stains_blend_alpha: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional

    """

    def __init__(
        self,
        stains_type="random",
        stains_blend_method="darken",
        stains_blend_alpha=0.5,
        p=1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.stains_type = stains_type
        self.stains_blend_method = stains_blend_method
        self.stains_blend_alpha = stains_blend_alpha

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"stains(stains_type={self.stains_type}, stains_blend_method={self.stains_blend_method}, stains_blend_alpha={self.stains_blend_alpha}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # convert and make sure image is color image
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            ysize, xsize = image.shape[:2]

            stains = ["rough_stains", "fine_stains", "severe_stains", "light_stains"]
            if self.stains_type not in stains:
                stains_type = random.choice(stains)
            else:
                stains_type = self.stains_type

            # generator to create stains
            stains_generator = TextureGenerator()

            # generate stains
            image_stains = stains_generator(
                texture_type=stains_type,
                texture_width=xsize,
                texture_height=ysize,
                quilt_texture=0,
            )

            # blend points image into input again
            ob = OverlayBuilder(
                self.stains_blend_method,
                image_stains,
                image,
                1,
                (1, 1),
                "center",
                0,
                self.stains_blend_alpha,
            )
            image_output = ob.build_overlay()

            # return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image_output] + outputs_extra
            else:
                return image_output

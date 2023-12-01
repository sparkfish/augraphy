import random

from augraphy.augmentations.lowinkline import LowInkLine


class LowInkRandomLines(LowInkLine):
    """Adds low ink lines randomly throughout the image.

    :param count_range: Pair of ints determining the range from which the number
           of lines is sampled.
    :type count_range: tuple, optional
    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        count_range=(5, 10),
        use_consistent_lines=True,
        noise_probability=0.1,
        p=1,
    ):
        """Constructor method"""
        super().__init__(
            use_consistent_lines=use_consistent_lines,
            noise_probability=noise_probability,
            p=p,
        )
        self.count_range = count_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkRandomLines(count_range={self.count_range}, use_consistent_lines={self.use_consistent_lines}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            image = image.copy()
            count = random.randint(self.count_range[0], self.count_range[1])

            for i in range(count):
                if image.shape[0] - 1 >= 1:
                    image = self.add_transparency_line(
                        image,
                        random.randint(1, image.shape[0] - 1),
                    )

            # check for additional output of mask, keypoints and bounding boxes
            outputs_extra = []
            if mask is not None or keypoints is not None or bounding_boxes is not None:
                outputs_extra = [mask, keypoints, bounding_boxes]

            # returns additional mask, keypoints and bounding boxes if there is additional input
            if outputs_extra:
                # returns in the format of [image, mask, keypoints, bounding_boxes]
                return [image] + outputs_extra
            else:
                return image

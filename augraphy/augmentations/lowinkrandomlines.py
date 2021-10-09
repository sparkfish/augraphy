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
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        count_range=(5, 10),
        use_consistent_lines=True,
        p=1,
    ):
        """Constructor method"""
        super().__init__(use_consistent_lines=use_consistent_lines, p=p)
        self.count_range = count_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkRandomLines(count_range={self.count_range}, use_consistent_lines={self.use_consistent_lines}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            count = random.randint(self.count_range[0], self.count_range[1])

            for i in range(count):
                if image.shape[0] - 1 >= 1:
                    image = self.add_transparency_line(
                        image,
                        random.randint(1, image.shape[0] - 1),
                    )

            return image

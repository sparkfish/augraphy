import random

from augraphy.base.augmentationresult import AugmentationResult
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

    def __init__(self, count_range=(5, 10), use_consistent_lines=True, p=0.5):
        """Constructor method"""
        super().__init__(use_consistent_lines=use_consistent_lines, p=p)
        self.count_range = count_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkRandomLines(count_range={self.count_range}, use_consistent_lines={self.use_consistent_lines}, p={self.p})"

    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            mask = data["ink"][-1].result.copy()
            count = random.randint(self.count_range[0], self.count_range[1])

            for i in range(count):
                mask = self.add_transparency_line(
                    mask, random.randint(1, mask.shape[0] - 1)
                )

            data["ink"].append(AugmentationResult(self, mask))

import random

from augraphy.augmentations.lowinkline import LowInkLine


class LowInkPeriodicLines(LowInkLine):
    """Creates a set of lines that repeat in a periodic fashion throughout the
    image.

    :param count_range: Pair of ints determining the range from which to sample
           the number of lines to apply.
    :type count_range: tuple, optional
    :param period_range: Pair of ints determining the range from which to sample
           the distance between lines.
    :type period_range: tuple, optional
    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        count_range=(2, 5),
        period_range=(10, 30),
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
        self.period_range = period_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkPeriodicLines(count_range={self.count_range}, period_range={self.period_range}, use_consistent_lines={self.use_consistent_lines}, p={self.p})"

    def add_periodic_transparency_line(self, mask, line_count, offset, alpha):
        """Creates horizontal lines of some opacity over the input image, at y-positions determined by the offset and line_count.

        :param mask: The image to apply the line to.
        :type mask: numpy.array
        :param line_count: The number of lines to generate.
        :type line_count: int
        :param offset: How far from the edge of the image to generate lines.
        :type offset: int
        :param alpha: The opacity of the lines.
        :type alpha: int
        """
        period = mask.shape[0] // line_count

        for y in range(mask.shape[0] - offset):
            if (period != 0) and (
                y % period == 0
            ):  # period can't be zero here, else there would be zero division error
                self.add_transparency_line(mask, y + offset, alpha)

    def add_periodic_transparency_lines(self, mask, lines, line_periods):
        """Creates horizontal lines of random opacity over the input image, at
        random intervals.

        :param mask: The image to apply the line to.
        :type mask: numpy.array
        :param lines: How many lines to add to the image.
        :type lines: int
        :param line_periods: The distance between lines.
        :type line_periods: int
        """
        period = mask.shape[0] // line_periods
        self.add_periodic_transparency_line(
            mask,
            line_periods,
            offset=random.randint(0, 5),
            alpha=random.randint(96, 255),
        )

        for i in range(lines):
            self.add_periodic_transparency_line(
                mask,
                line_periods,
                offset=random.randint(0, period),
                alpha=random.randint(16, 96),
            )

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()
            count = random.randint(self.count_range[0], self.count_range[1])
            period = random.randint(self.period_range[0], self.period_range[1])

            for i in range(count):
                self.add_periodic_transparency_lines(image, count, period)

            return image

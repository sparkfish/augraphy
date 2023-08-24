from augraphy.base.augmentation import Augmentation


class AugmentationSequence(Augmentation):
    """A list of Augmentations to be applied sequentially.

    :param augmentations: A list of Augmentation objects to be applied.
    :type augmentations: list
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, augmentations, p=1):
        """Constructor method"""
        self.augmentations = augmentations
        self.p = p
        self.results = []

    def __len__(self):
        return len(self.augmentations)

    def __repr__(self):
        output = "AugmentationSequence([\n"
        for aug in self.augmentations:
            output += f"\t{repr(aug)},\n"

        output += "])"
        return output

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        if force or self.should_run():
            result = image
            for augmentation in self.augmentations:
                if isinstance(result, tuple):
                    result = result[0]
                current_result = augmentation(result, mask=mask, keypoints=keypoints, bounding_boxes=bounding_boxes)

                if isinstance(augmentation, Augmentation):
                    if (mask is not None) or (keypoints is not None) or (bounding_boxes is not None):
                        current_result, mask, keypoints, bounding_boxes = current_result
                self.results.append(current_result)

                # make sure result is not None when parsing it to the next augmentation
                if not isinstance(result, tuple) and current_result is not None:
                    result = current_result
                elif isinstance(current_result, tuple):
                    if current_result[0] is not None:
                        result = current_result

            return result, self.augmentations

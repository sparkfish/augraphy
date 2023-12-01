class AugmentationResult:
    """Contains the result of an Augmentation's application, as well as
    the Augmentation applied. AugmentationResults are stored in an AugmentationPipeline.

    :param augmentation: The augmentation that was applied.
    :type augmentation: object
    :param result: The image transformed by the augmentation. Usually a numpy array.
    :type result: object
    :param mask: The mask of labels for each pixel. Mask value should be in range of 0 to 255.
    :type mask: numpy array (uint8), optional
    :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate (x, y).
    :type keypoints: dictionary, optional
    :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
    :type bounding_boxes: list, optional
    :param metadata: Additional data that may be added by callers.
    :type metadata: object, optional
    """

    def __init__(self, augmentation, result, mask=None, keypoints=None, bounding_boxes=None, metadata=None):
        """Constructor method"""
        self.augmentation = augmentation
        self.result = result
        self.mask = mask
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.metadata = metadata

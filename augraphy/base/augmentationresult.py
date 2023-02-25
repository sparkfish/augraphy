class AugmentationResult:
    """Contains the result of an Augmentation's application, as well as
    the Augmentation applied. AugmentationResults are stored in an AugmentationPipeline.

    :param augmentation: The augmentation that was applied.
    :type augmentation: object
    :param result: The image transformed by the augmentation. Usually a numpy array.
    :type result: object
    :param metadata: Additional data that may be added by callers.
    :type metadata: object, optional
    """

    def __init__(self, augmentation, result, metadata=None):
        """Constructor method"""
        self.augmentation = augmentation
        self.result = result
        self.metadata = metadata

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Function(Augmentation):
    """Accepts an arbitrary function or list of functions to apply in the pipeline.

    :param fs: The function(s) to apply.
    :type fs: function or list of functions
    """

    def __init__(self, fs, p=1):
        self.fs = fs
        super().__init__(p=p)

    def applyFs(self, fs, img):
        """Applies any fs to img sequentially."""
        current = img

        if type(fs) == list:
            for f in fs:
                current = f(current)

        else:
            current = fs(current)

        return current

    def __call__(self, image, force=False):
        image = image.copy()
        output = self.applyFs(self.fs, image)

        return output

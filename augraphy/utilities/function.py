from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Function(Augmentation):
    """Accepts an arbitrary function or list of functions to apply in the pipeline.

    :param fs: The function(s) to apply.
    :type fs: function
    :param kwargs: Arguments to the function.
    :type fs: argument name and value
    """

    def __init__(self, fs, p=1, **kwargs):
        self.fs = fs
        self.kwargs = kwargs
        super().__init__(p=p)

    def applyFs(self, fs, img):
        """Applies any fs to img sequentially."""

        img = fs(img, **self.kwargs)

        return img

    def __call__(self, image, layer=None, force=False):
        image = image.copy()
        output = self.applyFs(self.fs, image)

        return output

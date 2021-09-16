from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Function(Augmentation):
    """Accepts an arbitrary function or list of functions to apply in the pipeline.

    :param fs: The function(s) to apply.
    :type fs: function or list of functions
    :param layer: The layer this augmentation is applied in.
    :type layer: string
    """

    def __init__(self, fs, layer, p=0.5):
        self.fs = fs
        self.layer = layer
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

    def __call__(self, data, force=False):
        image = data[self.layer][-1].result.copy()
        output = self.applyFs(self.fs, image)

        data[self.layer].append(AugmentationResult(self, output))

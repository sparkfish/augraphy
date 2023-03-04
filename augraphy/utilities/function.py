from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Function(Augmentation):
    """Accepts an arbitrary function or list of functions to apply in the pipeline.

    :param fs: The function(s) to apply.
    :type fs: function or list of functions
    :param nkwargs: Arguments to the function(s).
    :type nkwargs: Dictionary list of dictionaries
    """

    def __init__(self, fs, nkwargs={}, p=1):
        self.fs = fs
        self.nkwargs = nkwargs
        super().__init__(p=p)

    def apply_functions(self, fs, image):
        """Applies any functions to image sequentially."""

        if type(fs) == list:
            for i, f in enumerate(fs):
                # check if the size of arguments is the same with functions or not
                if i <= len(self.nkwargs) - 1:
                    # not empty argument
                    if self.nkwargs[i]:
                        image = f(image, **self.nkwargs[i])
                    else:
                        image = f(image)
                # no argument are provided
                else:
                    image = f(image)
        else:
            image = fs(image, **self.nkwargs)

        return image

    def __call__(self, image, layer=None, force=False):
        image = image.copy()
        output = self.apply_functions(self.fs, image)

        return output

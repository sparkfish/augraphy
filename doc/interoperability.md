# Interoperability
You may wish to use Augraphy with other projects, such as [Albumentations](https://github.com/albumentations-team/albumentations), [imgaug](https://github.com/aleju/imgaug), or [imagecorruptions](https://github.com/bethgelab/imagecorruptions).

We support wrapping Albumentations augmentations and imgaug augmenters with the `Interop` class, defined in `src/augraphy/utilities/interop.py`. `imgaug` already provides wrapper classes for `imagecorruptions` transforms, so transitively we support those too. Transformations from other libraries may also work, but haven't been tested.

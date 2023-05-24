"""
version: 0.0.1


Dependencies

- opencv

*********************************

References:


- Numba Documentation: https://numba.readthedocs.io/en/stable/

- OpenCV Documentation:  https://docs.opencv.org/4.x/


"""
import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.detectdpi import DPIMetrics

# list which contains the possible dimensions of scanned pages in inches and their respective dpi (dots per inch)


class Rescale(Augmentation):
    def __init__(self, scale=None, target_dpi=300, p=1.0):
        """
        Rescale the image to the desired output

        :param image_path: list of path of images inside a directory
        :type image_path: array (String)
        :param targets:
        :type targets:
        :param resize:
        :type resize:
        """
        super().__init__(p=p)

        self.scale = scale

        self.target_dpi = target_dpi

    def _dpi_resize(self, image, doc_dimensions, target_dpi=300):

        width_inches, height_inches = doc_dimensions[0], doc_dimensions[1]

        width = width_inches * target_dpi

        height = height_inches * target_dpi

        output_image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)

        return output_image

    def __call__(self, image, layer=None, force=None, doc_dims=(None, None), original_dpi=None):

        if force or self.should_run():

            new_img = None
            if self.scale == "optimal":  # rescaling to user defined dpi before passing the img to augmentation pipeline
                obj = DPIMetrics(image)
                original_dpi, doc_dimensions = obj()
                if original_dpi != self.target_dpi:

                    new_img = self._dpi_resize(image=image, doc_dimensions=doc_dimensions, target_dpi=self.target_dpi)

                return {
                    "original_dpi": original_dpi,
                    "doc_dimensions": doc_dimensions,
                    "rescaled_img": new_img,
                    "output_dpi": self.target_dpi,
                }

            if self.scale == "original":
                new_img = self._dpi_resize(image=image, doc_dimensions=doc_dims, target_dpi=original_dpi)

                return new_img

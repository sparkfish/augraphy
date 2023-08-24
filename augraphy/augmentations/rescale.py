"""
version: 0.0.1


Dependencies

- opencv

*********************************

References:

- OpenCV Documentation:  https://docs.opencv.org/4.x/


"""
import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.utilities.detectdpi import dpi_resize
from augraphy.utilities.detectdpi import DPIMetrics


class Rescale(Augmentation):
    """Rescales image to the desired DPI for augmentation purpose.

    :param target_dpi: Value of the desired DPI.
    :type target_dpi: string, optional
    """

    def __init__(
        self,
        target_dpi=300,
        p=1.0,
    ):
        super().__init__(p=p)
        """Constructor method"""
        self.target_dpi = target_dpi

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Rescale(target_dpi={self.target_dpi}, p={self.p})"

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):

        if force or self.should_run():

            dpi_object = DPIMetrics(image)
            original_dpi, doc_dimensions = dpi_object()
            image_resize = dpi_resize(image=image, doc_dimensions=doc_dimensions, target_dpi=self.target_dpi)

            return image_resize

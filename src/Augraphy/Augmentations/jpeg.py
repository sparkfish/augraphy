################################################################################
# File: jpeg.py
#
# Class: JpegAugmentation
#
# Description: This file contains a class defining an Augmentation which uses
#              JPEG encoding to create JPEG compression artifacts in the image.
################################################################################



################################################################################
# Imports
################################################################################

import cv2
import numpy.random



################################################################################
# Class Definition
################################################################################

class JpegAugmentation(Augmentation):
  def __init__(self, quality_range=(25, 95), probability=0.5):
    super().__init__(probability=probability)
    self.quality_range = quality_range

  # Constructs a string representation of this Augmentation.
  def __repr__(self):
    return f"JpegAugmentation(quality_range={self.quality_range}, probability={self.probability})"

  # Applies the Augmentation to input data.
  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['post'][-1].result
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.quality_range[0], self.quality_range[1])]
      result, encimg = cv2.imencode('.jpg', image, encode_param)
      image = cv2.imdecode(encimg, 1)
      data['post'].append(AugmentationResult(self, image))

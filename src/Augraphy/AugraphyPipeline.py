import random
import cv2
import numpy as np

from Augraphy.PaperFactory import PaperFactory
from Augraphy.ImageTransformer import ImageTransformer
from Augraphy.Augmentations.PrinterAugmentation import PrinterAugmentation
from Augraphy.Augmentations.ScannerAugmentation import ScannerAugmentation

class AugraphyPipeline(ImageTransformer):
  def __init__(self, debug = False):
    super().__init__(debug)
    self.paper_factory = PaperFactory(debug=debug)

  def rotate_image(self, mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    mat = cv2.bitwise_not(mat)
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    rotated_mat = cv2.bitwise_not(rotated_mat)
    return rotated_mat

  def crappify(self, image, rotate=True):
    #scale = random.uniform(1.0, 2.0)

    printer = PrinterAugmentation(debug=self.debug)
    scanner = ScannerAugmentation(
                add_dirty_rollers = random.choice([True, False]),
                add_lighting_shadow = random.choice([True, False]),
                subtle_noise_range = random.randint(2,5),
                debug=self.debug)
    
    processed_image = self.transform(cv2.cvtColor, image, cv2.COLOR_BGR2GRAY)

    if (rotate):
      rotate_angle = random.uniform(-90,90)
      image = self.transform(self.rotate_image, image, rotate_angle)
      processed_image = self.transform(self.rotate_image, processed_image, rotate_angle)

    paper = self.transform(self.paper_factory.get_texture, processed_image.shape)
    processed_image = self.transform(printer, processed_image, paper)
    processed_image = self.transform(scanner, processed_image)
    
    #processed_image = cv2.resize(processed_image, (int(processed_image.shape[1] * scale), int(processed_image.shape[0] * scale)))
    #clean_image = cv2.resize(image, (int(processed_image.shape[1] * 2), int(processed_image.shape[0] * 2)))

    return processed_image, image
  
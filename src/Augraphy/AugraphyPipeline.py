import random
import cv2

from Augraphy.PaperFactory import PaperFactory
from Augraphy.ImageTransformer import ImageTransformer
from Augraphy.Augmentations.PrinterAugmentation import PrinterAugmentation
from Augraphy.Augmentations.ScannerAugmentation import ScannerAugmentation

class AugraphyPipeline(ImageTransformer):
  def __init__(self, debug = False):
    super().__init__(debug)
    self.paper_factory = PaperFactory(debug=debug)

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
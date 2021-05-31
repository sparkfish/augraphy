import random
import numpy as np
import cv2

from Augraphy.ImageTransformer import ImageTransformer
from Augraphy.Augmentations.InkBleedAugmentation import InkBleedAugmentation
from Augraphy.Augmentations.DustyInkAugmentation import DustyInkAugmentation
from Augraphy.Augmentations.LowInkAugmentation import LowInkAugmentation

class PrinterAugmentation(ImageTransformer):
  def __init__(self, debug=False):
    super().__init__(debug=debug)
    self.ink_bleed = InkBleedAugmentation(.2, debug=debug)
    self.dusty_ink = DustyInkAugmentation(.2, debug=debug)
    self.low_ink = LowInkAugmentation(
                          use_periodic_lines = random.choice([True, False]),
                          use_consistent_lines = random.choice([True, False]), debug=debug)
    
  def blur(self, img, kernel_size):
    blurr_options = ["gaussian", "average"]
    blurr_opt = random.choice(blurr_options)

    if blurr_opt == "average":
      img = cv2.blur(img,kernel_size)
    elif blurr_opt == "gaussian":
      img = cv2.GaussianBlur(img,kernel_size,0)

    return img

  def __call__(self, image, paper):
    ink = image

    if (self.debug or random.choice([True, False])):
      ink = self.transform(self.ink_bleed, ink)

    if (self.debug or random.choice([True, False])):
      ink = self.transform(self.dusty_ink, ink)

    ink = self.transform(self.make_white_transparent, ink, ink_color=random.randint(0, 96))

    if (self.debug or random.choice([True, False])):
      ink = self.transform(self.low_ink, ink)

    ink = self.transform(self.blur, ink, (3,3))

    return self.print_ink_to_paper(paper, ink)

  def make_white_transparent(self, img, ink_color=0):
    # Create the Ink Layer for the specified color.
    img_bgra = cv2.cvtColor(np.full((img.shape[0], img.shape[1], 3), ink_color, dtype="uint8"), cv2.COLOR_BGR2BGRA)

    # Convert to grayscale if not already.
    if (len(img.shape) > 2 and img.shape[2] > 1):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply transparency mask based on grayscale.
    img_bgra[:,:,3] = ~img[:,:]
    return img_bgra
    
  def print_ink_to_paper(self, background, overlay):
    # Split out the transparency mask from the colour info
    overlay_img = overlay[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    background_part = (background * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0))
  
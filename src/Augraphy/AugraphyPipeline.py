import random
import cv2
import numpy as np

from Augraphy.PaperFactory import PaperFactory

class AugraphyPipeline:
  def __init__(self, ink_phase, paper_phase, post_phase, paper_factory=None, ink_color_range=(0, 96), debug=False):
    self.ink_phase = ink_phase
    self.paper_phase = paper_phase
    self.post_phase = post_phase
    self.ink_color_range = ink_color_range

    if (paper_factory is None):
      self.paper_factory = PaperFactory()
    else:
      self.paper_factory = paper_factory

  def augment(self, image):
    ink = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = self.ink_phase(ink)
    ink = self.make_white_transparent(ink, random.randint(self.ink_color_range[0], self.ink_color_range[1]))
    paper = self.paper_factory.create_paper(image.shape)
    paper = self.paper_phase(paper)
    result = self.print_ink_to_paper(ink, paper)
    result = self.post_phase(result)

    return result.astype("uint8")

  def print_ink_to_paper(self, overlay, background):
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

  def make_white_transparent(self, img, ink_color=0):
    # Create the Ink Layer for the specified color.
    img_bgra = cv2.cvtColor(np.full((img.shape[0], img.shape[1], 3), ink_color, dtype="uint8"), cv2.COLOR_BGR2BGRA)

    # Convert to grayscale if not already.
    if (len(img.shape) > 2 and img.shape[2] > 1):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply transparency mask based on grayscale.
    img_bgra[:,:,3] = ~img[:,:]
    return img_bgra

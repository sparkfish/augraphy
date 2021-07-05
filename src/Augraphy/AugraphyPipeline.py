import random
import cv2
import numpy as np

class AugmentationResult():
  def __init__(self, augmentation, result, metadata=None):
    self.augmentation = augmentation
    self.result = result
    self.metadata = metadata

class AugraphyPipeline:
  def __init__(self, ink_phase, paper_phase, post_phase, ink_color_range=(0, 96), paper_color_range=(164, 255), rotate_range=(0,360)):
    self.ink_phase = ink_phase
    self.paper_phase = paper_phase
    self.post_phase = post_phase
    self.ink_color_range = ink_color_range
    self.rotate_range = rotate_range
    self.paper_color_range = paper_color_range

  def augment(self, image):
    data = dict()
    data['image'] = image.copy()
    ink = data['image'].copy()
    ink = self.rotate_image(ink, random.randint(self.rotate_range[0], self.rotate_range[1]))
    data['image_rotated'] = ink.copy()
    
    if (len(ink.shape) > 2 and ink.shape[2] == 3):
      ink = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
    elif (len(ink.shape) > 2 and ink.shape[2] == 4):
      ink = cv2.cvtColor(ink, cv2.COLOR_BGRA2GRAY)
    
    data['pipeline'] = self
    data['ink'] = list()
    data['paper'] = list() 
    data['post'] = list()

    data['ink'].append(AugmentationResult(None, ink))
    data['paper'].append(AugmentationResult(None, np.full((ink.shape[0], ink.shape[1], 3), random.randint(self.paper_color_range[0],self.paper_color_range[1]), dtype="uint")))


    self.ink_phase(data)
    self.paper_phase(data)

    data['post'].append(AugmentationResult(None, self.print_ink_to_paper(data['ink'][-1].result, data['paper'][-1].result)))

    self.post_phase(data)

    data['output'] = data['post'][-1].result.astype("uint8")

    return data

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

  def print_ink_to_paper(self, overlay, background):
    overlay = self.make_white_transparent(overlay, random.randint(self.ink_color_range[0], self.ink_color_range[1]))
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

  def __repr__(self):
    r = f"ink_phase = {repr(self.ink_phase)}\n\n"
    r += f"paper_phase = {repr(self.ink_phase)}\n\n"
    r += f"post_phase = {repr(self.ink_phase)}\n\n"

    r += f"AugraphyPipeline(ink_phase, paper_phase, post_phase, ink_color_range={self.ink_color_range}, paper_color_range={self.paper_color_range}, rotate_range={self.rotate_range})"

    return r

  def visualize(self):
    print(repr(self))


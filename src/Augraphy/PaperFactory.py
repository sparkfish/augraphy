import random
import cv2
import numpy as np
from glob import glob

from Augraphy.ImageTransformer import ImageTransformer

class PaperFactory(ImageTransformer):
  def __init__(self, use_texture=True, use_generated=True, texture_path="./paper_textures", debug=False):
    super().__init__(debug=debug)
    self.create_methods = list()

    if (use_texture):
      self.create_methods.append(self.get_texture)
      self.paper_textures = list()
      
      for file in glob(f"{texture_path}/*"):
        self.paper_textures.append(cv2.imread(file))

    if (use_generated):
      self.create_methods.append(self.generate_paper)

  def create_paper(self, shape):
    return random.choice(self.create_methods)(shape)

  def generate_paper(self, shape):
    paper = np.full((shape[0], shape[1], 3), random.randint(160, 255), dtype="uint8")

    if (self.debug or random.choice([True, False])):
      paper = self.transform(self.texture, paper)

    if (self.debug or random.choice([True, False])):
      paper = self.transform(self.make_paper, paper, 0.9, 0.99, .03)

    if (self.debug or random.choice([True, False])):
      blur_kernel = (random.choice([3, 5]), random.choice([3, 5]))
      paper = self.transform(self.blur, paper, blur_kernel)
    
    return paper

  def brightness(self, img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

  def make_paper(self, img, low, high, deviation):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)

    low = value-(value*deviation) # *random.uniform(0, deviation)
    max = value+(value*deviation)
    brightness_matrix = np.random.uniform(low, max, (hsv.shape[0], hsv.shape[1]))
    hsv[:,:,1] *= brightness_matrix
    hsv[:,:,2] *= brightness_matrix
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2][hsv[:,:,2]>255]  = 255

    hsv = np.array(hsv, dtype = np.uint8)
    hsv = cv2.bitwise_not(hsv)
    hsv = np.array(hsv, dtype = np.float64)

    low = value-(value*deviation)
    max = value+(value*deviation)
    brightness_matrix = np.random.uniform(low, max, (hsv.shape[0], hsv.shape[1]))
    hsv[:,:,1] *= brightness_matrix
    hsv[:,:,2] *= brightness_matrix
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2][hsv[:,:,2]>255]  = 255

    hsv = np.array(hsv, dtype = np.uint8)
    hsv = cv2.bitwise_not(hsv)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

  def texture(self, image, sigma=5, turbulence=2):
      """
      Consequently applies noise patterns to the original image from big to small.

      sigma: defines bounds of noise fluctuations
      turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
      value - the more iterations will be performed during texture generation.
      """
      result = image.astype(float)
      cols, rows, ch = image.shape
      ratio = cols
      while not ratio == 1:
          result += self.noise(cols, rows, ratio, sigma=sigma)
          ratio = (ratio // turbulence) or 1
      cut = np.clip(result, 0, 255)
      return cut.astype(np.uint8)

  def noise(self, width, height, ratio=1, sigma=5):
      """
      The function generates an image, filled with gaussian nose. If ratio parameter is specified,
      noise will be generated for a lesser image and then it will be upscaled to the original size.
      In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
      uses interpolation.

      :param ratio: the size of generated noise "pixels"
      :param sigma: defines bounds of noise fluctuations
      """
      mean = 0
      #assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
      #assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

      h = int(height / ratio)
      w = int(width / ratio)

      if (h == 0):
        h = 1
      if (w == 0):
        w = 1

      result = np.random.normal(mean, sigma, (w, h, 1))
      if ratio > 1:
        result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
      return result.reshape((width, height, 1))
  
  def get_texture(self, shape):
    texture = random.choice(self.paper_textures)
    texture_h = texture.shape[0]
    texture_w = texture.shape[1]
    shape_h = shape[0]
    shape_w = shape[1]

    if (texture_h > shape_h or texture_w > shape_w): # Zoom out
      h_ratio = shape_h / texture_h
      w_ratio = shape_w / texture_w

      if (h_ratio > w_ratio):
        scale = random.uniform(h_ratio, 1.2)
      else:
        scale = random.uniform(w_ratio, 1.2)
      
      zoom = (int(texture_w*scale), int(texture_h*scale))
      #print(f"Zoom out from {texture.shape} to {zoom}")
      texture = cv2.resize(texture, zoom)
      texture_h = texture.shape[0]
      texture_w = texture.shape[1]

    if (texture_h <= shape_h or texture_w <= shape_w): # Zoom in
      h_ratio = shape_h / texture_h
      w_ratio = shape_w / texture_w

      if (h_ratio > w_ratio):
        scale = random.uniform(h_ratio, h_ratio+1.5)
      else:
        scale = random.uniform(w_ratio, w_ratio+1.5)
      zoom = (int(texture_w*scale), int(texture_h*scale))
      #print(f"Zoom in from {texture.shape} to {zoom}")
      texture = cv2.resize(texture, zoom)

    #if (random.choice([True, False])):
    texture = self.brightness(texture, 0.8, 0.99)
    
    h = random.randint(0, texture.shape[0]-shape[0])
    w = random.randint(0, texture.shape[1]-shape[1])
    cropped_texture = texture[h:h+shape[0], w:w+shape[1]]

    return cropped_texture
  
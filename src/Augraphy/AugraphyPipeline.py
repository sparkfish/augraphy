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

  def white_balance(self, img):
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    k = (r_avg + g_avg + b_avg) / 3
    kr = (k / r_avg) #+ random.uniform(-0.1,0.1) 
    kg = (k / g_avg) #+ random.uniform(-0.1,0.1)
    kb = (k / b_avg) + random.uniform(-0.05,0.05)
    
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    imbalance_img = cv2.merge([b, g, r])

    return imbalance_img

  def rotate(self, img, angle):
    h, w = img.shape[:2]
    img = cv2.bitwise_not(img)
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    img = cv2.bitwise_not(img)
    return img

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
  
  def blur(self, img, kernel_size):
    blurr_options = ["gaussian", "average"]
    blurr_opt = random.choice(blurr_options)

    if blurr_opt == "average":
      img = cv2.blur(img,kernel_size)
    elif blurr_opt == "gaussian":
      img = cv2.GaussianBlur(img,kernel_size,0)

    return img


  #def add_defects(img):

  def streaks(self, img):
    h, w = img.shape[:2]
    mask = np.zeros((h,w,3),dtype=np.uint8)

    num_defects = random.randint(100,150)

    for i in range(num_defects):
      x1 = random.randint(0,w-30)
      y1 = random.randint(0,h-30)
      x_diff = random.randint(1,20)
      y_diff = random.randint(1,50)
      cv2.line(img,(x1,y1),(x1,y1+y_diff),(255,255,255), 1)

    alpha=0
    #cv2.imwrite("mask.png",mask)
    img = img+mask
    return img

  def transparent_streaks(self, img):
    h, w = img.shape[:2]
    #mask = np.zeros((h,w,3),dtype=np.uint8)

    num_defects = random.randint(100,150)

    for i in range(num_defects):
      x1 = random.randint(0,w-30)
      y1 = random.randint(0,h-30)
      x_diff = random.randint(1,20)
      y_diff = random.randint(1,50)

      cv2.line(img,(x1,y1),(x1,y1+y_diff),(255,255,255,0), 1)

    alpha=0
    #cv2.imwrite("mask.png",mask)
    #img = img+mask
    return img


  def wspot(self, img):
    h, w = img.shape[:2]
    mask = np.zeros((h,w,3),dtype=np.uint8)

    num_defects = random.randint(100,150)

    for i in range(num_defects):
      x1 = random.randint(0,w-30)
      y1 = random.randint(0,h-30)
      x_diff = random.randint(1,4)
      y_diff = random.randint(1,4)
      cv2.line(img,(x1,y1),(x1,y1+y_diff),(255,255,255), 1)

    alpha=0
    cv2.imwrite("mask.png",mask)
    img = img+mask
    return img

  def noisy(self, image, noise_typ):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 3 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

  def add_noise(self, img, sigma=5):
      """
      Adds noise to the existing image
      """
      width, height, ch = img.shape
      n = self.noise(width, height, sigma=sigma)
      img = img + n
      return img.clip(0, 255)

  def crappify_barcode(self, image, **kwargs):
    return self.crappify_barcode_v2(image, **kwargs)

  def crappify_barcode_v2(self, image, add_scanlines=True, add_line_noise=True, blur=True, parallel_light=True, random_paper_color=True, texturize_paper=True, rotate=True):
    scale = random.uniform(1.0, 2.0)

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
    processed_image = cv2.resize(processed_image, (int(processed_image.shape[1] * scale), int(processed_image.shape[0] * scale)))
    clean_image = cv2.resize(image, (int(processed_image.shape[1] * 2), int(processed_image.shape[0] * 2)))

    return processed_image, clean_image
  
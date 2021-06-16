import types
import math
import cv2
import numpy as np
import random

from glob import glob
from scipy.stats import norm
from sklearn.datasets import make_blobs

try:
  from google.colab.patches import cv2_imshow
  IN_COLAB = True
except:
  IN_COLAB = False

class Augmentation():
  def __init__(self, probability=0.5, debug = False, debug_slant_perspective = False):
    self.debug = debug  
    self.probability = probability
    self.debug_slant_perspective = debug_slant_perspective

  def _imshow(self, name, image):
    if (self.debug):
      image = image.astype("uint8")
      # if (len(image.shape) > 2 and image.shape[2] == 4):
      #   image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
      if (self.debug_slant_perspective):
        image = self.slant_perspective(image)
      
      if (IN_COLAB):
        print(name)
        cv2_imshow(image)
      else:
        cv2.imshow(name, image)

  def transform(self, transform, *args, **kwargs):
    result = transform(*args, **kwargs)

    if (isinstance(transform, types.BuiltinFunctionType) or isinstance(transform, types.MethodType)):
      self._imshow(transform.__name__, result)
    else:
      self._imshow(transform.__class__.__name__, result)
      
    return result
  
  def blur(self, img, kernel_size):
    blurr_options = ["gaussian", "average"]
    blurr_opt = random.choice(blurr_options)

    if blurr_opt == "average":
      img = cv2.blur(img,kernel_size)
    elif blurr_opt == "gaussian":
      img = cv2.GaussianBlur(img,kernel_size,0)

    return img

  def sobel(self, image):
      gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
      gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
      gradient = cv2.subtract(gradX, gradY)
      gradient = cv2.convertScaleAbs(gradient)
      return gradient

  def grayscale(self, image):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  
  def threshold(self, *kargs, **kwargs):
    (_, thresh) = cv2.threshold(*kargs, **kwargs)
    return thresh

  def slant_perspective(self, img):
    img = self.add_border(img)
    warp_pixels = img.shape[1] // 2
    src = np.array([[0,0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
    dst = np.array([[warp_pixels,0], [img.shape[1] + warp_pixels, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
    warped = self.warp(img, src, dst)
    return warped

  def warp(self, img, src, dst):
    """
    Args:
        img: np.array
        src: list
        dst: list
    Returns:
        un_warped: np.array
    """
    # calculate the tranformation
    h, w = img.shape[:2]
    mat = cv2.getPerspectiveTransform(src.astype("float32"), dst.astype("float32"))

    # new source: image corners
    corners = np.array([
                    [0, h],
                    [0, 0],
                    [w, 0],
                    [w, h]
                ])

    # Transform the corners of the image
    corners_tranformed = cv2.perspectiveTransform(
                                  np.array([corners.astype("float32")]), mat)

    x_mn = math.ceil(min(corners_tranformed[0].T[0]))
    y_mn = math.ceil(min(corners_tranformed[0].T[1]))

    x_mx = math.ceil(max(corners_tranformed[0].T[0]))
    y_mx = math.ceil(max(corners_tranformed[0].T[1]))

    new_w = x_mx - x_mn
    new_h = y_mx - y_mn
    dst2 = corners_tranformed
    dst2 -= np.array([x_mn, y_mn])

    H = cv2.getPerspectiveTransform(corners.astype("float32"), dst2.astype("float32"))
    #H, _ = cv2.findHomography(src, dst2, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if (len(img.shape) > 2 and img.shape[2] == 3):
      border_value = (255,255,255)
    elif (len(img.shape) > 2 and img.shape[2] == 4):
      border_value = (255,255,255,255)
    else:
      border_value = (255,)

    un_warped = cv2.warpPerspective(img, H, (new_w, new_h), 
                              flags=cv2.INTER_LINEAR,
                              borderMode = cv2.BORDER_CONSTANT, 
                              borderValue = border_value)


    return un_warped

  def add_border(self, img):
    row, col = img.shape[:2]
    bottom = img[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 5
    if (len(img.shape) > 2 and img.shape[2] == 3):
      border_value = [0,0,0]
    elif (len(img.shape) > 2 and img.shape[2] == 4):
      border_value = [0,0,0,0]
    else:
      border_value = [0,]

    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return border

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
  
  def should_run(self): return random.uniform(0.0, 1.0) <= self.probability

class AugmentationSequence(Augmentation):
  def __init__(self, augmentations, probability=1.0, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.augmentations = augmentations

  def __call__(self, img, force=False):
    if (force or self.should_run()):
      for augmentation in self.augmentations:
        img = self.transform(augmentation, img)
    
    return img

class GaussianBlurAugmentation(Augmentation):
  def __init__(self, kernels=[(3,3)], sigmaX=0, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.sigmaX = sigmaX
    self.kernels = kernels

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      image = cv2.GaussianBlur(image, random.choice(self.kernels), self.sigmaX)

    return image

class OneOf(Augmentation):
    def __init__(self, augmentations, probability=0.5, debug=False):
        super().__init__(probability=probability, debug=debug)
        self.augmentations = augmentations
        augmentation_probabilities = [augmentation.probability for augmentation in augmentations]
        s = sum(augmentation_probabilities)
        self.augmentation_probabilities = [ap / s for ap in augmentation_probabilities]

    def __call__(self, image, force=False):
        if self.augmentation_probabilities and (force or self.should_run()):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            augmentation = random_state.choice(self.augmentations, p=self.augmentation_probabilities)
            image = self.transform(augmentation, image, force=True)

        return image

class BrightnessTexturizeAugmentation(Augmentation):
  def __init__(self, range=(0.9, 0.99), deviation=0.03, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.low = range[0]
    self.high = range[1]
    self.deviation = deviation

  def __call__(self, img, force=False):
    if (force or self.should_run()):
      value = random.uniform(self.low, self.high)
      hsv = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_BGR2HSV)
      hsv = np.array(hsv, dtype = np.float64)

      low = value-(value*self.deviation) # *random.uniform(0, deviation)
      max = value+(value*self.deviation)
      brightness_matrix = np.random.uniform(low, max, (hsv.shape[0], hsv.shape[1]))
      hsv[:,:,1] *= brightness_matrix
      hsv[:,:,2] *= brightness_matrix
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2][hsv[:,:,2]>255]  = 255

      hsv = np.array(hsv, dtype = np.uint8)
      hsv = cv2.bitwise_not(hsv)
      hsv = np.array(hsv, dtype = np.float64)

      low = value-(value*self.deviation)
      max = value+(value*self.deviation)
      brightness_matrix = np.random.uniform(low, max, (hsv.shape[0], hsv.shape[1]))
      hsv[:,:,1] *= brightness_matrix
      hsv[:,:,2] *= brightness_matrix
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2][hsv[:,:,2]>255]  = 255

      hsv = np.array(hsv, dtype = np.uint8)
      hsv = cv2.bitwise_not(hsv)
      img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

class NoiseTexturizeAugmentation(Augmentation):
  """
      Consequently applies noise patterns to the original image from big to small.

      sigma: defines bounds of noise fluctuations
      turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
      value - the more iterations will be performed during texture generation.
  """

  def __init__(self, sigma_range=(3, 10), turbulence_range=(2, 5), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.sigma_range = sigma_range
    self.turbulence_range = turbulence_range

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      sigma = random.randint(self.sigma_range[0], self.sigma_range[1])
      turbulence = random.randint(self.turbulence_range[0], self.turbulence_range[1])

      result = image.astype(float)
      cols, rows, ch = image.shape
      ratio = cols
      while not ratio == 1:
          result += self.noise(cols, rows, ratio, sigma=sigma)
          ratio = (ratio // turbulence) or 1
      cut = np.clip(result, 0, 255)
      return cut.astype(np.uint8)

    return image

  def noise(self, width, height, ratio, sigma):
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

class DirtyRollersAugmentation(Augmentation):
  def __init__(self, line_width_range=(8, 12), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.line_width_range = line_width_range

  def apply_scanline_mask(self, img, mask, meta_mask):
    if (random.choice([True, False])):
      return self.apply_scanline_mask_v2(img, mask, meta_mask)
    else:
      return self.apply_scanline_mask_v1(img, mask, meta_mask)

  def apply_scanline_mask_v2(self, img, mask, meta_mask):
    mask = self.transform(self.apply_scanline_metamask_v2, mask, meta_mask)
    update_lambda = lambda x, y: min(255, x + (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_metamask_v2(self, img, mask):
    update_lambda = lambda x, y: max(0, x - (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_mask_v1(self, img, mask, meta_mask):
    mask = self.transform(self.apply_scanline_metamask_v1, mask, meta_mask)
    update_lambda = lambda x, y: max(0, x - (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def apply_scanline_metamask_v1(self, img, mask):
    update_lambda = lambda x, y: min(99, x + (x * (1-(y/100))))
    update = np.vectorize(update_lambda)
    return update(img, mask)

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      line_width = random.randint(self.line_width_range[0], self.line_width_range[1])
      rotate = random.choice([True, False])

      if (not self.debug and rotate):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      mask = self.transform(self.create_scanline_mask, image.shape[1], image.shape[0], line_width)
      meta_mask = self.transform(self.create_scanline_mask, image.shape[1], image.shape[0], line_width * random.randint(10, 25))
      image = self.transform(self.apply_scanline_mask, image, mask, meta_mask).astype("uint8")
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

      if (not self.debug and rotate):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

  def create_scanline_mask(self, width, height, line_width):
    grad_list = list()

    # Create Standard Bar
    grad_high_pct = random.randint(86, 99)
    grad_low_pct = random.randint(70,85)

    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Wide Dark
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_center = np.full((random.randint(1, 6)), grad_low_pct)
    grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Wide Light
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)
    
    # Create Standard Bar with Lower Dark
    grad_high_pct += min(100, random.randint(-3, 3))
    grad_low_pct -= random.randint(5, 8)
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Low Dark and Wide Dark
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_center = np.full((random.randint(1, 6)), grad_low_pct)
    grad_grid = np.hstack((grad_grid, grad_center, np.flip(grad_grid)))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    # Create Standard Bar with Low Dark Wide Light
    grad_dec = (grad_low_pct-grad_high_pct)/(line_width)
    grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
    grad_exterior = np.full((random.randint(1, 6)), grad_high_pct)
    grad_grid = np.hstack((grad_grid, np.flip(grad_grid), grad_exterior))
    grad_grid = np.tile(grad_grid,(height,1))
    grad_list.append(grad_grid)

    mask = random.choice(grad_list)
    while (mask.shape[1] < width):
      mask = np.hstack((mask, random.choice(grad_list)))

    return mask[:,0:width]

class LowInkAugmentation(Augmentation):
  def __init__(self, mask_augmentations, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.mask_augmentations = AugmentationSequence(mask_augmentations, debug=debug)
    transparency_lambda = lambda x, y: int((255-y)/255.0 * x) #~y if x != 0 and y != 0 else x
    self.apply_mask = np.vectorize(transparency_lambda)

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      rotate = random.choice([True, False])

      if (rotate):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

      mask = np.full((image.shape[0], image.shape[1]), 0, dtype="uint8")
      mask = self.transform(self.mask_augmentations, mask)
      image[:,:,3] = self.apply_mask(image[:,:,3], mask)

      if (rotate):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

class LowInkBlobsAugmentation(Augmentation):
  def __init__(self, count_range=(5, 25), size_range=(10, 20), points_range=(5, 25), std_range=(10, 75), features_range=(15, 25), value_range=(180, 250), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.count_range = count_range
    self.size_range = size_range
    self.points_range = points_range
    self.std_range = std_range
    self.features_range = features_range
    self.value_range = value_range
    apply = lambda x, y: x if x > y else y
    self.apply = np.vectorize(apply)
  
  def create_blob(self):
    size = random.randint(self.size_range[0], self.size_range[1])
    std = random.randint(self.std_range[0], self.std_range[1]) / 100
    points = random.randint(self.points_range[0], self.points_range[1])
    features = random.randint(self.features_range[0], self.features_range[1])
    
    X, y = make_blobs(n_samples=points, cluster_std=[std], centers=[(0, 0)], n_features=features) #, random_state=1)
    X *= (size // 4)
    X += (size // 2)
    X = [[int(item) for item in items] for items in X]
    blob = np.full((size, size, 1), 0, dtype="uint8")

    for point in X:
      if point[0] < blob.shape[0] and point[1] < blob.shape[1] and point[0] > 0 and point[1] > 0:
        value = random.randint(self.value_range[0], self.value_range[1])
        blob[point[0], point[1]] = value
        
    return blob
  
  def apply_blob(self, mask):
    blob = self.create_blob()
    x_start = random.randint(0, mask.shape[1] - blob.shape[1])
    y_start = random.randint(0, mask.shape[0] - blob.shape[0])
    x_stop = x_start + blob.shape[1]
    y_stop = y_start + blob.shape[0]
    mask_chunk = mask[y_start:y_stop, x_start:x_stop]
    mask[y_start:y_stop, x_start:x_stop] = self.apply(mask_chunk, blob[:,:,0])

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      count = random.randint(self.count_range[0], self.count_range[1])

      for i in range(count):
        self.apply_blob(image)

    return image

class LowInkLineAugmentation(Augmentation):  
  def __init__(self, use_consistent_lines=True, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    
    self.use_consistent_lines = use_consistent_lines
    inconsistent_transparency_line = lambda x: random.randint(0, 255)
    self.inconsistent_transparency_line = np.vectorize(inconsistent_transparency_line)

    apply_line = lambda x, y: x if x > y else y
    self.apply_line = np.vectorize(apply_line)

  def add_transparency_line(self, mask, y, alpha=None):
    if (alpha == None):
      alpha = random.randint(16, 224)

    if (self.use_consistent_lines):
      low_ink_line = np.full(mask[y, :].shape, alpha, dtype="uint8")
    else:
      low_ink_line = self.inconsistent_transparency_line(mask[y, :])

    mask[y, :] = self.apply_line(mask[y, :], low_ink_line)

    return mask

class LowInkRandomLinesAugmentation(LowInkLineAugmentation):
  def __init__(self, count_range=(5, 10), use_consistent_lines=True, probability=0.5, debug=False):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability, debug=debug)
    self.count_range = count_range
  
  def __call__(self, mask, force=False):
    if (force or self.should_run()):
      count = random.randint(self.count_range[0], self.count_range[1])

      for i in range(count):
        mask = self.add_transparency_line(mask, random.randint(1, mask.shape[0]-1))

    return mask

class LowInkPeriodicLinesAugmentation(LowInkLineAugmentation):
  def __init__(self, count_range=(5, 10), period_range=(10, 30), use_consistent_lines=True, probability=0.5, debug=False):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability, debug=debug)
    self.count_range = count_range
    self.period_range = period_range

  def add_periodic_transparency_line(self, mask, line_count, offset, alpha):
    period = mask.shape[0] // line_count
    
    for y in range(mask.shape[0]-offset):
      if (y % period == 0):
        self.add_transparency_line(mask, y+offset, alpha)

  def add_periodic_transparency_lines(self, mask, lines, line_periods):
    period = mask.shape[0] // line_periods
    self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, 5), alpha=random.randint(96, 255))

    for i in range(lines):
      self.add_periodic_transparency_line(mask, line_periods, offset=random.randint(0, period), alpha=random.randint(16, 96))

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      count = random.randint(self.count_range[0], self.count_range[1])
      period = random.randint(self.period_range[0], self.period_range[1])

      for i in range(count):
        self.add_periodic_transparency_lines(image, count, period)

    return image

class DustyInkAugmentation(Augmentation):
  def __init__(self, intensity_range=(.1, .2), color_range=(0, 224), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.intensity_range = intensity_range
    self.color_range = color_range


  def __call__(self, img, force=False):
    if (force or self.should_run()):
      intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
      add_noise_fn = lambda x: random.randint(self.color_range[0], self.color_range[1]) if (x == 0 and random.random() < intensity) else x
      add_noise = np.vectorize(add_noise_fn)
      return add_noise(img)

    return img
  
class InkBleedAugmentation(Augmentation):
  def __init__(self, intensity_range=(.1, .2), color_range=(0, 224), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.intensity_range = intensity_range
    self.color_range = color_range

  def __call__(self, img, force=False):
    if (force or self.should_run()):
      intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
      add_noise_fn = lambda x, y: random.randint(self.color_range[0], self.color_range[1]) if (y == 255 and random.random() < intensity) else x
      add_noise = np.vectorize(add_noise_fn)
      sobel = self.transform(self.sobel, img)
      return add_noise(img, sobel)
    
    return img

class SubtleNoiseAugmentation(Augmentation):
  def __init__(self, range=5, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.range = range
    self.add_subtle_noise = np.vectorize(lambda x: max(0, min(255, x + random.randint(-self.range, self.range))))

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      image = self.add_subtle_noise(image)

    return image

class JpegAugmentation(Augmentation):
  def __init__(self, quality_range=(50, 95), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.quality_range = quality_range

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.quality_range[0], self.quality_range[1])]
      result, encimg = cv2.imencode('.jpg', image, encode_param)
      image = cv2.imdecode(encimg, 1)

    return image

class BrightnessAugmentation(Augmentation):
  def __init__(self, range=(0.8, 1.4), probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.range = range

  def __call__(self, img, force=False):
    if (force or self.should_run()):
      value = random.uniform(self.range[0], self.range[1])
      hsv = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_BGR2HSV)
      hsv = np.array(hsv, dtype = np.float64)
      hsv[:,:,1] = hsv[:,:,1]*value
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2] = hsv[:,:,2]*value
      hsv[:,:,2][hsv[:,:,2]>255]  = 255
      hsv = np.array(hsv, dtype = np.uint8)
      img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

class LightingGradientAugmentation(Augmentation):
  """
    Generate decayed light mask generated by light strip given its position, direction
    Args:
      mask_size: tuple of integers (w, h) defining generated mask size
      position: tuple of integers (x, y) defining the center of light strip position,
        which is the reference point during rotating
      direction: integer from 0 to 360 to indicate the rotation degree of light strip
      max_brightness: integer that max brightness in the mask
      min_brightness: integer that min brightness in the mask
      mode: the way that brightness decay from max to min: linear or gaussian
      linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]
    Return:
      light_mask: ndarray in float type consisting value from 0 to strength
  """

  def __init__(self, light_position=None, direction=None, max_brightness=255, min_brightness=0, mode="gaussian", linear_decay_rate=None, transparency=None, probability=0.5, debug=False):
    super().__init__(probability=probability, debug=debug)
    self.light_position = light_position
    self.direction = direction
    self.max_brightness = max_brightness
    self.min_brightness = min_brightness
    self.mode = mode
    self.linear_decay_rate = linear_decay_rate
    self.transparency = transparency

  def __call__(self, image, force=False):
    if (force or self.should_run()):
      if self.transparency is None:
        transparency = random.uniform(0.5, 0.85)
      else:
        transparency = self.transparency

      frame = image
      height, width, _ = frame.shape
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      mask = self.generate_parallel_light_mask(mask_size=(width, height), position=self.light_position, direction=self.direction, max_brightness=self.max_brightness, min_brightness=self.min_brightness, mode=self.mode, linear_decay_rate=self.linear_decay_rate)
      hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
      frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      frame[frame > 255] = 255
      frame = np.asarray(frame, dtype=np.uint8)
      return frame

    return image

  def generate_parallel_light_mask(self, mask_size, position=None, direction=None, max_brightness=255, min_brightness=0, mode="gaussian", linear_decay_rate=None):
    if position is None:
      pos_x = random.randint(0, mask_size[0])
      pos_y = random.randint(0, mask_size[1])
    else:
      pos_x = position[0]
      pos_y = position[1]
    if direction is None:
      direction = random.randint(0, 360)
    if linear_decay_rate is None:
      if mode == "linear_static":
        linear_decay_rate = random.uniform(0.2, 2)
    if mode == "linear_dynamic":
        linear_decay_rate = (max_brightness - min_brightness) / max(mask_size)
    assert mode in ["linear_dynamic", "linear_static", "gaussian"], \
      "mode must be linear_dynamic, linear_static or gaussian"
    padding = int(max(mask_size) * np.sqrt(2))
    # add padding to satisfy cropping after rotating
    canvas_x = padding * 2 + mask_size[0]
    canvas_y = padding * 2 + mask_size[1]
    mask = np.zeros(shape=(canvas_y, canvas_x), dtype=np.float32)
    # initial mask's up left corner and bottom right corner coordinate
    init_mask_ul = (int(padding), int(padding))
    init_mask_br = (int(padding+mask_size[0]), int(padding+mask_size[1]))
    init_light_pos = (padding + pos_x, padding + pos_y)
    # fill in mask row by row with value decayed from center
    for i in range(canvas_y):
      if mode == "linear":
        i_value = self._decayed_value_in_linear(i, max_brightness, init_light_pos[1], linear_decay_rate)
      elif mode == "gaussian":
        i_value = self._decayed_value_in_norm(i, max_brightness, min_brightness, init_light_pos[1], mask_size[1])
      else:
        i_value = 0
      mask[i] = i_value
    # rotate mask
    rotate_M = cv2.getRotationMatrix2D(init_light_pos, direction, 1)
    mask = cv2.warpAffine(mask, rotate_M, (canvas_x,  canvas_y))
    # crop
    mask = mask[init_mask_ul[1]:init_mask_br[1], init_mask_ul[0]:init_mask_br[0]]
    mask = np.asarray(mask, dtype=np.uint8)
    # add median blur
    mask = cv2.medianBlur(mask, 9)
    mask = 255 - mask
    # cv2.circle(mask, init_light_pos, 1, (0, 0, 255))
    # cv2.imshow("crop", mask[init_mask_ul[1]:init_mask_br[1], init_mask_ul[0]:init_mask_br[0]])
    # cv2.imshow("all", mask)
    # cv2.waitKey(0)
    return mask  
    
  def _decayed_value_in_norm(self, x, max_value, min_value, center, range):
    """
    decay from max value to min value following Gaussian/Normal distribution
    """
    radius = range / 3
    center_prob = norm.pdf(center, center, radius)
    x_prob = norm.pdf(x, center, radius)
    x_value = (x_prob / center_prob) * (max_value - min_value) + min_value
    return x_value

  def _decayed_value_in_linear(self, x, max_value, padding_center, decay_rate):
    """
    decay from max value to min value with static linear decay rate.
    """
    x_value = max_value - abs(padding_center - x) * decay_rate
    if x_value < 0:
      x_value = 1
    return x_value

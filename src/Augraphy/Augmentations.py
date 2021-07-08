################################################################################
# File: Augmentations.py
#
# Classes: Augmentation
#          AugmentationResult
#          AugmentationSequence(Augmentation)
#          OneOf(Augmentation)
#
# Description: This file contains classes defining the base Augmentation,
#              a sequence of Augmentations, a randomly selected Augmentation,
#              and the result of an application of an Augmentation.
################################################################################


################################################################################
# Imports
################################################################################

import cv2
import numpy as np
import random
import os

from glob import glob
from scipy.stats import norm
from sklearn.datasets import make_blobs


################################################################################
# Class Definitions
################################################################################

# The base Augmentation class which everything else inherits.
class Augmentation:
    def __init__(self, probability=0.5):
        self.probability = probability

    # Determines whether or not the augmentation should be
    # applied by callers.
    def should_run(self):
        return random.uniform(0.0, 1.0) <= self.probability


# Contains the result of an Augmentation's application.
class AugmentationResult:
    def __init__(self, augmentation, result, metadata=None):
        self.augmentation = augmentation
        self.result = result
        self.metadata = metadata



# Contains a list of Augmentations to be applied.
class AugmentationSequence(Augmentation):
    def __init__(self, augmentations, probability=1.0):
        super().__init__(probability=probability)
        self.augmentations = augmentations

  def __repr__(self):
    return f"LowInkLineAugmentation(use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

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
  def __init__(self, count_range=(5, 10), use_consistent_lines=True, probability=0.5):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability)
    self.count_range = count_range

  def __repr__(self):
    return f"LowInkRandomLinesAugmentation(count_range={self.count_range}, use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"
  
  def __call__(self, data, force=False):
    if (force or self.should_run()):
      mask = data['ink'][-1].result.copy()
      count = random.randint(self.count_range[0], self.count_range[1])

      for i in range(count):
        mask = self.add_transparency_line(mask, random.randint(1, mask.shape[0]-1))

      data['ink'].append(AugmentationResult(self, mask))

class LowInkPeriodicLinesAugmentation(LowInkLineAugmentation):
  def __init__(self, count_range=(2, 5), period_range=(10, 30), use_consistent_lines=True, probability=0.5):
    super().__init__(use_consistent_lines=use_consistent_lines, probability=probability)
    self.count_range = count_range
    self.period_range = period_range

  def __repr__(self):
    return f"LowInkPeriodicLinesAugmentation(count_range={self.count_range}, period_range={self.period_range}, use_consistent_lines={self.use_consistent_lines}, probability={self.probability})"

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

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['ink'][-1].result.copy()
      count = random.randint(self.count_range[0], self.count_range[1])
      period = random.randint(self.period_range[0], self.period_range[1])

      for i in range(count):
        self.add_periodic_transparency_lines(image, count, period)

      data['ink'].append(AugmentationResult(self, image))

class DustyInkAugmentation(Augmentation):
  def __init__(self, intensity_range=(.1, .2), color_range=(0, 224), probability=0.5):
    super().__init__(probability=probability)
    self.intensity_range = intensity_range
    self.color_range = color_range

  def __repr__(self):
    return f"DustyInkAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, probability={self.probability})"


  def __call__(self, data, force=False):
    if (force or self.should_run()):
      img = data['ink'][-1].result
      intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
      add_noise_fn = lambda x: random.randint(self.color_range[0], self.color_range[1]) if (x == 0 and random.random() < intensity) else x
      add_noise = np.vectorize(add_noise_fn)
      img = add_noise(img)

      data['ink'].append(AugmentationResult(self, img))
  
class InkBleedAugmentation(Augmentation):
  def __init__(self, intensity_range=(.1, .2), color_range=(0, 224), probability=0.5):
    super().__init__(probability=probability)
    self.intensity_range = intensity_range
    self.color_range = color_range

  def __repr__(self):
    return f"InkBleedAugmentation(intensity_range={self.intensity_range}, color_range={self.color_range}, probability={self.probability})"

  def sobel(self, image):
      gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
      gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
      gradient = cv2.subtract(gradX, gradY)
      gradient = cv2.convertScaleAbs(gradient)
      return gradient

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      img = data['ink'][-1].result
      intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
      add_noise_fn = lambda x, y: random.randint(self.color_range[0], self.color_range[1]) if (y == 255 and random.random() < intensity) else x
      add_noise = np.vectorize(add_noise_fn)
      sobel = self.sobel(img)
      img = add_noise(img, sobel)

      data['ink'].append(AugmentationResult(self, img))

class SubtleNoiseAugmentation(Augmentation):
  def __init__(self, range=10, probability=0.5):
    super().__init__(probability=probability)
    self.range = range
    self.add_subtle_noise = np.vectorize(lambda x: max(0, min(255, x + random.randint(-self.range, self.range))))

  def __repr__(self):
    return f"SubtleNoiseAugmentation(range={self.range}, probability={self.probability})"

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['post'][-1].result
      data['post'].append(AugmentationResult(self, self.add_subtle_noise(image)))

class JpegAugmentation(Augmentation):
  def __init__(self, quality_range=(25, 95), probability=0.5):
    super().__init__(probability=probability)
    self.quality_range = quality_range

  def __repr__(self):
    return f"JpegAugmentation(quality_range={self.quality_range}, probability={self.probability})"

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['post'][-1].result
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.quality_range[0], self.quality_range[1])]
      result, encimg = cv2.imencode('.jpg', image, encode_param)
      image = cv2.imdecode(encimg, 1)
      data['post'].append(AugmentationResult(self, image))

class BrightnessAugmentation(Augmentation):
  def __init__(self, layer, range=(0.8, 1.4), probability=0.5):
    super().__init__(probability=probability)
    self.range = range
    self.layer = layer

  def __repr__(self):
    return f"BrightnessAugmentation({self.layer}, range={self.range}, probability={self.probability})"

  def __call__(self, data, force=False):
    if (force or self.should_run()):

      img = data[self.layer][-1].result
      value = random.uniform(self.range[0], self.range[1])
      if self.layer=="ink":
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
      hsv = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_BGR2HSV)

      hsv = np.array(hsv, dtype = np.float64)
      hsv[:,:,1] = hsv[:,:,1]*value
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2] = hsv[:,:,2]*value
      hsv[:,:,2][hsv[:,:,2]>255]  = 255
      hsv = np.array(hsv, dtype = np.uint8)
      img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      if self.layer == "ink":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      data[self.layer].append(AugmentationResult(self, img))
class GammaAugmentation(Augmentation):
  def __init__(self, range=(0.5, 1.5), probability=0.5):
    super().__init__(probability=probability)
    self.range = range

  def __repr__(self):
    return f"GammaAugmentation(range={self.range}, probability={self.probability})"

  def __call__(self, data,force=False):
    img = data['post'][-1].result
    img = img.astype(np.uint8)
    value = random.uniform(self.range[0], self.range[1])
    invGamma = 1.0 / value
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    print(table.shape,img.shape)
    frame= cv2.LUT(img, table)
    data['post'].append(AugmentationResult(self, frame))
    # cv2.imwrite(os.path.join('test_outputs',str(round(value,4))+"gamma.jpg"),np.hstack((img,frame)))


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

  def __init__(self, light_position=None, direction=None, max_brightness=255, min_brightness=0, mode="gaussian", linear_decay_rate=None, transparency=None, probability=0.5):
    super().__init__(probability=probability)
    self.light_position = light_position
    self.direction = direction
    self.max_brightness = max_brightness
    self.min_brightness = min_brightness
    self.mode = mode
    self.linear_decay_rate = linear_decay_rate
    self.transparency = transparency

  def __repr__(self):
    return f"LightingGradientAugmentation(light_position={self.light_position}, direction={self.direction}, max_brightness={self.max_brightness}, min_brightness={self.min_brightness}, mode='{self.mode}', linear_decay_rate={self.linear_decay_rate}, transparency={self.transparency}, probability={self.probability})"

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      image = data['post'][-1].result
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
      data['post'].append(AugmentationResult(self, frame))

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

class PaperFactory(Augmentation):
  def __init__(self, tile_texture_shape=(250,250), texture_path="./paper_textures", probability=0.5):
    super().__init__(probability=probability)
    self.paper_textures = list()
    self.tile_texture_shape = tile_texture_shape
    self.texture_path = texture_path
    for file in glob(f"{texture_path}/*"):
      texture = cv2.imread(file)

      if (len(texture.shape) > 2 and texture.shape[2] == 4):
        texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2BGR)
      elif (len(texture.shape) > 2 and texture.shape[2] == 3):
        pass
      else:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

      self.paper_textures.append(cv2.imread(file))

  def __repr__(self):
    return f"PaperFactory(tile_texture_shape={self.tile_texture_shape}, texture_path={self.texture_path}, probability={self.probability})"

  def __call__(self, data, force=False):
    if (force or self.should_run()):
      shape = data['ink'][-1].result.shape

      if (random.choice([True, False])):
        texture = self.get_texture(self.tile_texture_shape)
        paper = self.tile_texture(texture, shape)
      else:
        texture = self.get_texture(shape)
        paper = texture.copy()

      data['paper_texture'] = texture
      data['paper'].append(AugmentationResult(self, paper))

  def tile_texture(self, texture, shape):
    x_scale = shape[0] // texture.shape[0] + 1
    y_scale = shape[1] // texture.shape[1] + 1
  
    if (len(texture.shape) > 2):
      paper = np.empty((texture.shape[0]*x_scale, texture.shape[1]*y_scale, texture.shape[2]))
    else:
      paper = np.empty((texture.shape[0]*x_scale, texture.shape[1]*y_scale))
    
    for x in range(x_scale):
      for y in range(y_scale):
        start_x = x*texture.shape[0]
        end_x = start_x+texture.shape[0]
        start_y = y*texture.shape[1]
        end_y = start_y+texture.shape[1]

        paper[start_x:end_x,start_y:end_y] = texture
        texture = cv2.flip(texture, 1)

      texture = cv2.flip(texture, 0)
      if (x_scale % 2 == 0):
        texture = cv2.flip(texture, 1)

    return paper[:shape[0], :shape[1]]

  def resize(self, texture, shape):
    texture_h = texture.shape[0]
    texture_w = texture.shape[1]
    shape_h = shape[0]
    shape_w = shape[1]

    if (texture_h > shape_h or texture_w > shape_w): # Zoom out
      h_ratio = shape_h / texture_h
      w_ratio = shape_w / texture_w

    # The entire sequence can be applied with this.
    def __call__(self, data, force=False):
        if force or self.should_run():
            for augmentation in self.augmentations:
                augmentation(data)

    # Constructs a string containing the representations
    # of each augmentation in the sequence.
    def __repr__(self):
        r = f"AugmentationSequence([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], probability={self.probability})"
        return r


# Given a list of Augmentations, selects one to apply.
class OneOf(Augmentation):
    def __init__(self, augmentations, probability=0.5):
        super().__init__(probability=probability)
        self.augmentations = augmentations

        # Compute the average probability from all augmentations.
        augmentation_probabilities = [
            augmentation.probability for augmentation in augmentations
        ]
        s = sum(augmentation_probabilities)
        self.augmentation_probabilities = [ap / s for ap in augmentation_probabilities]

    # Randomly selects an Augmentation to apply to data.
    def __call__(self, data, force=False):
        if self.augmentation_probabilities and (force or self.should_run()):

            # Seed the random object.
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            # Randomly selects one Augmentation to apply.
            augmentation = random_state.choice(
                self.augmentations, p=self.augmentation_probabilities
            )

            # Applies the selected Augmentation.
            augmentation(data, force=True)

    # Constructs a string containing the representations
    # of each augmentation
    def __repr__(self):
        r = f"OneOf([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], probability={self.probability})"
        return r

################################################################################
# File: brightness.py
#
# Class: BrightnessAugmentation
#
# Description: This file contains a class defining an Augmentation which adjusts
#              the brightness of the whole image by a chosen multiplier.
################################################################################



################################################################################
# Imports
################################################################################

import cv2
import numpy as np
import random



################################################################################
# Class Definition
################################################################################

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


  # Constructs a string representation of this Augmentation.
  def __repr__(self):
    return f"PaperFactory(tile_texture_shape={self.tile_texture_shape}, texture_path={self.texture_path}, probability={self.probability})"

  # Applies the Augmentation to input data.
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

  # Takes a texture and a shape and returns the shape with
  # the texture tiled in a vertical/horizontal grid across it. 
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

  # Scales and zooms a given texture to fit a given shape.
  def resize(self, texture, shape):
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

    return texture

  # Returns a paper texture cropped to a given shape.
  def get_texture(self, shape):
    texture = random.choice(self.paper_textures)

    if (texture.shape[0] < shape[0] or texture.shape[1] < shape[1]):
      texture = self.resize(texture, shape)

    h = random.randint(0, texture.shape[0]-shape[0])
    w = random.randint(0, texture.shape[1]-shape[1])
    cropped_texture = texture[h:h+shape[0], w:w+shape[1]]
    return cropped_texture

import random
import cv2
import numpy as np
from glob import glob

class PaperFactory:
  def __init__(self, texture_shape=(250,250), texture_path="./paper_textures"):
    self.paper_textures = list()
    self.texture_shape = texture_shape
    for file in glob(f"{texture_path}/*"):
      self.paper_textures.append(cv2.imread(file))

  def create_paper(self, shape):
    texture = self.get_texture(self.texture_shape)
    paper = self.tile(texture, shape)
    return paper

  def tile(self, texture, shape):
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

      if (x_scale % 2 == 1):
        texture = cv2.flip(texture, 1)
      texture = cv2.flip(texture, 0)

    return paper[:shape[0], :shape[1]]

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
  
  def get_texture(self, shape):
    texture = random.choice(self.paper_textures)
    h = random.randint(0, texture.shape[0]-shape[0])
    w = random.randint(0, texture.shape[1]-shape[1])
    cropped_texture = texture[h:h+shape[0], w:w+shape[1]]
    return cropped_texture

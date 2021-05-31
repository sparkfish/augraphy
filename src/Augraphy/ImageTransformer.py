import types
import math
import cv2
import numpy as np
import random

class ImageTransformer():
  def __init__(self, debug = False, debug_slant_perspective = False):
    self.debug = debug  
    self.debug_slant_perspective = debug_slant_perspective

  def _imshow(self, name, image):
    if (self.debug):
      image = image.astype("uint8")
      # if (len(image.shape) > 2 and image.shape[2] == 4):
      #   image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
      if (self.debug_slant_perspective):
        image = self.slant_perspective(image)
        
      print(name)
      cv2_imshow(image)

  def transform(self, transform, *args, **kwargs):
    result = transform(*args, **kwargs)

    print(str(transform))
    if (isinstance(transform, types.FunctionType)):
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
  
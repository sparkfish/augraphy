from Augraphy.AugraphyPipeline import AugraphyPipeline
import string
import random
import cv2
import numpy as np
import PIL
import os
from pdf417gen import encode, render_image

from Augraphy.AugraphyPipeline import AugraphyPipeline

def create_pdf417():
    type = 'pdf417'
    codeword_multiplier = random.randint(1, 9)
    text_length = codeword_multiplier * 50
    text = ''.join(random.choices(string.ascii_letters + string.digits + "&,:#-.$/+%* =^;<>@[\\]_'~!|()?{}", k=text_length))
    scale = random.randint(2, 3)
    padding = 5
    columns = codeword_multiplier + 2
    ratio = random.randint(2, 4)
    encoded = encode(text, columns=columns)
    image = render_image(encoded, scale=scale, ratio=ratio, padding=padding)

    return image

img = create_pdf417()
cv2.imshow("original", img)
cv2.waitKey(1000)
pipeline = AugraphyPipeline()
crappified = pipeline.crappify_barcode(img)

cv2.imshow("crappified", crappified)
cv2.waitKey(5000)

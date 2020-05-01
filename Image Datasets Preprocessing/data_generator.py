from PIL import Image
import os
import math
import cv2
import numpy as np
import glob
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def key_func(x):
    return os.path.split(x)[-1]


Count = 0
Train_Count = 0
Orig_Images = "Image/"
path = "Cropped/"

for filename in os.listdir(Orig_Images):
    im = Image.open(os.path.join(Orig_Images, filename))
    if im is not None:
        # Opens a image in RGB mode
        width, height = im.size
        width = math.ceil(width/32)
        height = math.ceil(height/32)
        # Setting the points for cropped image
        x1 = 0
        y1 = 0
        x2 = 32
        y2 = 32

        for i in range(0, height):

            for j in range(0, width):
                im1 = im.crop((x1, y1, x2, y2))
                f, e = os.path.splitext(path + 'Part ')
                Count = Count + 1
                im1.save(f + str(Count) + '.jpg', 'JPEG', quality=90)
                x1 = x1 + 32
                x2 = x2 + 32

            x1 = 0
            y1 = y1 + 32
            x2 = 32
            y2 = y2 + 32

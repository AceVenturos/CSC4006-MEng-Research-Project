#!/usr/bin/python
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys

path = os.getcwd() + "\\places365_standard128\\val\\"
dirs = os.listdir(path)

newDataset = os.getcwd() + "\\places365_standard64\\val\\"

def resize():
    for item in dirs:
        newPath = os.listdir(path + item)
        os.mkdir(newDataset + item + "\\")
        for image in newPath:
            #print(path + item + "\\" + image)
            if os.path.isfile(path + item + "\\" + image):
                im = Image.open(path + item + "\\" + image)
                f, e = os.path.splitext(newDataset + item + "\\" + image)
                imResize = im.resize((64, 64), Image.ANTIALIAS)
                print(f + '.jpg')
                imResize.save(f + '.jpg', 'JPEG', quality=90)


resize()

# Used to resize railroad_track directory as there was some issues with this (images missing, resized copy of directory
# from 'large' dataset - Jamie 13/01 17:32
# from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import os, sys
#
# path = os.getcwd()
# dirs = os.listdir(path)
#
# def resize():
#     for item in dirs:
#             im = Image.open(item)
#             f, e = os.path.splitext(item)
#             imResize = im.resize((128,128), Image.ANTIALIAS)
#             imResize.save(f + '.jpg', 'JPEG', quality=90)
#
# resize()

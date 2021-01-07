#!/usr/bin/python
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys

path = os.getcwd() + "\\temp\\"
dirs = os.listdir(path)

print(dirs)


def resize():
    for item in dirs:
        newPath = os.listdir(path + item)
        for image in newPath:
            print(path + item + "\\" + image)
            if os.path.isfile(path + item + "\\" + image):
                im = Image.open(path + item + "\\" + image)
                f, e = os.path.splitext(path + item + "\\" + image)
                imResize = im.resize((128, 128), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=90)


resize()

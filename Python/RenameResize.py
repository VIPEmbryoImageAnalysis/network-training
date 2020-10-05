#Renames and Resizes files in a given folder

from PIL import Image
import os, sys

w = 0; #how many frames before the frame you start renaming

path = '' #input path
files = os.listdir(path); #get the files

def resize():
    for item in files:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(f + ' resized.png', 'PNG', quality=90)

resize()

for i, filename in enumerate(os.listdir(path)):
    os.rename(" " + filename, " " + str(w) + ".png");


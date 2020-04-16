# Network Training Script

import os
import numpy as np
import glob
from PIL import Image
import mahotas as mh
import matplotlib as plt

# Define Image Locations
imgDir = "/This PC/Desktop/Embryo Extracted Frames"     # Location of extracted frames for training
labelDir = "/This PC/Desktop/Embryo Labeled Test"       # Location of labeled frames for training
for file_name in glob.iglob('/This PC/Desktop/Embryo Extracted Frames'):
    img = Image.open(file_name)     # Open images (frames) inside of inputted folder
    labels = [1,2]      # Define labels on frames (Label 1 = Background, Label 2 = Embryo)
    labelNames = ["Background", "Embryo"]       # Names of labels used

    





# IN PROGRESS

import os
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from config import imshape, model_name, n_classes
from models import preprocess_input, dice
from tensorflow.keras.utils import to_categorical
from utils import add_masks
import matplotlib.pyplot as plt
from train import sorted_fcn


files = []
pixelArray = []
timeArray = []
pixeltoreal = 0.58 * 0.58

# Load the trained model
model = load_model(os.path.join('models', 'unet_multi.model'),
                   custom_objects={'dice': dice})

# Ask how much videos is being analyzed
numVideos = int(input('How many videos are you processing? '))

# Ask for file input
# These files need to be inside the same folder
for x in range(1, numVideos+1):
    print('Video', x)
    n = input("What is the filepath? ")
    files += [n]
    
# Extract frames and save them
for j in files:
    temp_path = 'images/'
    video = cv2.VideoCapture('AsKiGPS1E11132.avi')
    success,image = video.read()
    videoNum = 1 #change video number here
    count = 1
    success = True
    while success:
        cv2.imwrite(os.path.join(temp_path , "video%d_frame%d.png" % (videoNum, count)), image)
        success,image = video.read()
        count += 1

# Iterate over frames


# perform prediciton
for x in frames:
    
    img_bgr = cv2.imread(#feed in image)
    Image.fromarray(img_bgr)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tmp = np.expand_dims(img, axis=0)
    roi_pred = model.predict(tmp)
    roi_mask = roi_pred.squeeze() * 255.0
    roi_mask = add_masks(roi_mask)
    roi_mask = np.array(roi_mask, dtype=np.uint8)
    roi_mask = cv2.addWeighted(img, 1.0, roi_mask, 1.0, 0)
    
    # save image to new folder
    
    video
    
# evaluate scores


import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from models import preprocess_input, dice
from tensorflow.keras.utils import to_categorical
from utils import add_masks
import matplotlib.pyplot as plt

imX = 256
imY = 256
totalPix =  imX  * imY

# Load the trained model
model = load_model(os.path.join('models', 'unet_multi.model'),
                   custom_objects={'dice': dice})

test_images_path = "/Users/shauncorpuz/Desktop/Micro VIP Embryo/TestingPy/Images/"
test_segs_path = "/Users/shauncorpuz/Desktop/Micro VIP Embryo/TestingPy/Labels/"

count = 1
# Loop through folders containing extracted image files
# and apply segmentation to them
for y in os.listdir(test_images_path):
    # Generate prediction mask
    img_bgr = cv2.imread(y)
    Image.fromarray(img_bgr)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tmp = np.expand_dims(img, axis=0)
    roi_pred = model.predict(tmp)
    roi_mask = roi_pred.squeeze() * 255.0
    roi_mask = add_masks(roi_mask)
        
    # Determine the number of predicted pixels
    pixelcount = 0
    for x in list(range(256)):
        for y in list(range(256)):
            if roi_mask[x,y,1] > 0:
                count += 1
            else:
                break
    
    # Take ratio of predicted embryo size to total pixels
    # Embryo Percentage
    emPerc = (pixelcount / totalPix) * 100
    # Background Percentage
    backPerc = ((totalPix - pixelcount) / totalPix) * 100
    
    # Information needed for plotting
    classes = ('Background', 'Embryo')
    y_pos = np.arange(len(classes))
    pixelperc = [backPerc, emPerc]

    # Plot the prediction plots
    plt.bar(y_pos, pixelperc, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    
    plt.show()

        
        
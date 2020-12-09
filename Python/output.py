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
import pytesseract
import csv
import matplotlib.pyplot as plt

files = []
org_folders = []
pixelArray = []
timeArray = []
pixeltoreal = 0.58 * 0.58
imsize = (256,256)
q = 1
homepath = os.getcwd()

# Create an empty CSV file
with open('embryo_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Videos','Time (h)','Initial Size (um^2)','Final Size (um^2)',
                   'Final Size Rank','Average Growth Rate','Avg Growth Rate Rank']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    thewriter.writeheader()
    

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
    
# Extract frames and save them in gray scale
for j in files:
    temp_path = homepath + '/%d' % (q) + 'images/'
    org_folders += temp_path
    video = cv2.VideoCapture(j)
    success,image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cropped using ROI coordinates
    # img[y coordinates, x coordinates]
    currentTime = image[473:486, 452:486]

    # use tesseracts' OCR function
    text = pytesseract.image_to_string(currentTime, lang='eng')

    # convert the string recieved from OCR to int with numbers only
    M = int(''.join(filter(str.isdigit, text)))

    # convert to realtime displayed on time stamp and store it
    extractedTime = M / 10
    timeArray.append(extractedTime)
    
    # resize image
    image = cv2.resize(image, imsize)
    
    count = 1
    success = True
    while success:
        cv2.imwrite(os.path.join(temp_path , "frame%d.png" % (count)), image)
        success,image = video.read()
        count += 1
    q += 1

count = 1
# Loop through folders containing extracted image files
# and apply segmentation to them
for x in org_folders:
    for y in os.listdir(x):
        
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
        
        # Store the size of embryo into an array
        pixelArray.append(pixelcount * pixeltoreal)
        
        # Generate overlay image of the prediction onto the original
        roi_mask = np.array(roi_mask, dtype=np.uint8)
        roi_mask = cv2.addWeighted(img, 1.0, roi_mask, 1.0, 0)
        
        # save image ** Will need to update to save in a new folder **
        cv2.imwrite(os.path.join(x , "Segmented_Frame_%d.png" % (count)), image)
        count += 1
        
# Linear Regression Plots
z = np.polyfit(timeArray, pixelArray, 1)
p = np.poly1d(z)

xp = np.linspace(timeArray[0], timeArray[-1], 100)
genplot = plt.plot(timeArray, pixelArray, '--', label='Real Time Growth')
plt.plot(xp, p(xp), '-', label='Average Growth Rate')
plt.xlabel('Time (Hours)')
plt.ylabel('Area (μm²)')
plt.title(name)
plt.legend()


# Initial Size
initSize = pixelArray[0]

# Final Size
finalSize = pixelArray[-1]

# Input Data Into CSV File
thewriter.writerow({'Videos':j, 'Time (h)':timeArray[-1], 'Initial Size (um^2)':pixelArray[0], 
                    'Final Size Rank':1, 'Average Growth Rate': p[1], 
                    'Avg Growth Rate Rank':1, 'Final Size (um^2)':pixelArray[-1]})


# If there are more than 1 video...
if numVideos > 1:
    
    # Reopen the CSV file
    with open('embryo_results.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
     
    # Size Ranking
    tempSize = []
    
    for i in data[1:]:
        tempSize.append(float((i[3])))
    
    x = tempSize
    index = [0]*len(x)

    for i in range(len(x)):
        index[x.index(min(x))] = i
        x[x.index(min(x))] = max(x)+1
    
    # Replace preset ranking with new ranking
    pos1 = 0
    for i in data[1:]:
        i[4] = str(index[pos1])
        pos1 = pos1 + 1
    
    # Growth Ranking
    tempGrowth = []
    for i in data[1:]:
        tempGrowth.append(int(float(i[5])))
    
    x = tempGrowth
    index = [0]*len(x)

    for i in range(len(x)):
        index[x.index(min(x))] = i
        x[x.index(min(x))] = max(x)+1
        
    # Replace preset ranking with new ranking
    pos2 = 0
    for i in data[1:]:
            i[6] = str(index[pos2])
            pos2 = pos2 + 1
    
    # Rewrite list into CSV
    with open('embryo_results.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in data:
            wr.writerow(i)
    



        

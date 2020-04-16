# script to find ROI coordinates

import numpy as np
import cv2

# function programmed to display coordinate on the image wherever clicked on
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' , ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 0.3, (255, 255, 0), 1)
        cv2.imshow('image', img)

# input image path
img = cv2.imread('/Users/shauncorpuz/Desktop/mouse4981/001_mouse4981.png')

# display image
cv2.imshow('image', img)

# when mouse is clicked, do the designated click event
cv2.setMouseCallback('image', click_event)

# press any button to terminate window when done
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)



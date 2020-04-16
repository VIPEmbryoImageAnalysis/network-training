# script that extracts the time stamp on an image

import cv2
import pytesseract

#Define path to images
#imagepath = "/Users/shauncorpuz/Desktop/mouse4981/001_mouse4981.png"
img = cv2.imread("/Users/shauncorpuz/Desktop/mouse4981/121_mouse4981.png")

# Cropped using ROI coordinates
# img[y coordinates, x coordinates]
currentTime = img[473:486, 452:486]

# use tesseracts' OCR function
text = pytesseract.image_to_string(currentTime, lang='eng')

# convert the string recieved from OCR to int with numbers only
M = int(''.join(filter(str.isdigit, text)))

# convert to realtime displayed on time stamp
extractedTime = M / 10

print(extractedTime)


# uncomment these lines if you want to see cropped image  
#cv2.imshow('image', currentTime)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)




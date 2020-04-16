import cv2
video = cv2.VideoCapture('D2019.07.10_S00701_I0746_D_wells_AA2(4981)_video.avi')
success,image = video.read()
videoNum = 4981 #change video number here
count = 1
success = True
while success:
    cv2.imwrite("video%d_frame%d.png" % (videoNum, count), image)
    success,image = video.read()
    count += 1
print('Number of frames read: ',count-1)

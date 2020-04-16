frameNum = '4321';
A = labeloverlay(imread(strcat('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (human and mice) r6\images\',frameNum,'.png')), ... 
    imread(strcat('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (human and mice) r6\labels\',frameNum,'.png')));
imshow(A)
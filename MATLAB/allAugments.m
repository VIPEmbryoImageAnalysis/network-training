%% Settings
%Modify lines 3-4, 8-9 as needed
isImage = 1; %0 for labels, 1 for original images
isGray = 1; %0 if not grayscale images, converts to gray scale (0 for LABELS), 1 for grayscale/original images
d90 = 90; d180 = 180; d270 = 270;%degrees, 90, 180, 270
widthlength = 496; %size of the resized image
K_p = 0;
originalpath = 'C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (human and mice) r3\base_images\'; %folder of desired frames to be augmented
newpath = 'C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (human and mice) r3\'; %location of output folder of augmented frames
%important notes:
%   only IMAGES are supposed to be grayscaled, LABELS are NOT grayscaled.
%   depending on total number of frames, edit the imwrite function at the
%   very bottom to make sure the naming convention of output files are
%   accurate.
%% Augments
type = 'Labels';
if(isImage == 1)
    type = 'Images';
end
files = dir(strcat(originalpath,'\*.png'));
typename = 'Label';
if(isImage==1)
    typename = 'Image';
end
N = strcat(newpath,'All_Augments_',typename);
lengthFile = length(files);
w = lengthFile;
mkdir(N);
%rotate 90
for k = 1:lengthFile
    Image = imread(strcat(originalpath,files(k).name));
    if (isImage==1 && isGray==0)
        Image = rgb2gray(Image);
    end
    Image = imrotate(Image,d90);
    if(isImage == 0)
            Image = label2rgb(Image);
            [r, c, ~] = size(Image);
            val1 = Image(1,1,1);
            val2 = Image(1,1,2);
            val3 = Image(1,1,3);
            for p = 1:r
                for l = 1:c
                    if(Image(p,l,1) == val1 && Image(p,l,2) == val2 && Image(p,l,3) == val3)
                        Image(p,l,1) = 0;
                        Image(p,l,2) = 0;
                        Image(p,l,3) = 255;
                    end
                end
            end
            [Image, map] = rgb2ind(Image, 2);
    end
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(Image,filename);
end
K_p = K_p + 1*lengthFile;

%rotate 180
for k = 1:length(files)
    Image = imread(strcat(originalpath,files(k).name));
    if (isImage==1 && isGray==0)
        Image = rgb2gray(Image);
    end
    Image = imrotate(Image,d180);
    if(isImage == 0)
            Image = label2rgb(Image);
            [r, c, ~] = size(Image);
            val1 = Image(1,1,1);
            val2 = Image(1,1,2);
            val3 = Image(1,1,3);
            for p = 1:r
                for l = 1:c
                    if(Image(p,l,1) == val1 && Image(p,l,2) == val2 && Image(p,l,3) == val3)
                        Image(p,l,1) = 0;
                        Image(p,l,2) = 0;
                        Image(p,l,3) = 255;
                    end
                end
            end
            [Image, map] = rgb2ind(Image, 2);
    end
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(Image,filename);
end
K_p = K_p + 1*lengthFile;

%rotate 270
for k = 1:length(files)
    Image = imread(strcat(originalpath,files(k).name));
    if (isImage==1 && isGray==0)
        Image = rgb2gray(Image);
    end
    Image = imrotate(Image,d270);
    if(isImage == 0)
            Image = label2rgb(Image);
            [r, c, ~] = size(Image);
            val1 = Image(1,1,1);
            val2 = Image(1,1,2);
            val3 = Image(1,1,3);
            for p = 1:r
                for l = 1:c
                    if(Image(p,l,1) == val1 && Image(p,l,2) == val2 && Image(p,l,3) == val3)
                        Image(p,l,1) = 0;
                        Image(p,l,2) = 0;
                        Image(p,l,3) = 255;
                    end
                end
            end
            [Image, map] = rgb2ind(Image, 2);
    end
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(Image,filename);
end
K_p = K_p + 1*lengthFile;

% mirror x
for k=1:lengthFile
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(Image,filename);
end
K_p = K_p + 1*lengthFile;

% mirror y
for k=1:lengthFile
    Image = imread(strcat(originalpath,files(k).name));
    Image = flip(Image,2); % flipping over y axis
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(Image,filename);
end
K_p = K_p + 1*lengthFile;

% combine
for k=1:lengthFile
    Image = imread(strcat(originalpath,files(k).name));
    nextImage = imread(strcat(originalpath,files(lengthFile).name));
    sizeImage = size(Image);
    totalpixels_x = sizeImage(2);
    totalpixels_y = sizeImage(1);
    segment_x = totalpixels_x/2;
    segment_y = totalpixels_y/2;
    if(lengthFile ~= k)
        I1(1:totalpixels_y,1:segment_x) = Image(1:totalpixels_y,1:segment_x);
        I1(1:totalpixels_y,segment_x+1:2*segment_x) = nextImage(1:totalpixels_y,segment_x+1:2*segment_x);
        I2(1:totalpixels_y,1:segment_x) = nextImage(1:totalpixels_y,1:segment_x);
        I2(1:totalpixels_y,segment_x+1:2*segment_x) = Image(1:totalpixels_y,segment_x+1:2*segment_x);
    end
    lengthFile = lengthFile-1;
    
   if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(I1,filename);
    w=w+1;
    
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    Image = imresize(Image, [widthlength widthlength]);
    imwrite(I2,filename);
end
lengthFile = length(files); %has to be re-established because of the mirror augment
K_p = K_p + 1*lengthFile;

% crop
for k=1:lengthFile
    Image = imread(strcat(originalpath,files(k).name));
    nextImage = imread(strcat(originalpath,files(lengthFile).name));
    sizeImage = size(Image);
    totalpixels_x = sizeImage(2);
    totalpixels_y = sizeImage(1);
    segment_x = totalpixels_x/2;
    segment_y = totalpixels_y/2;
    I1 = Image(1:segment_y,1:segment_x);
    I1 = imresize(I1,[widthlength,widthlength]);
    I2 = Image(1:segment_y,segment_x+1:2*segment_x);
    I2 = imresize(I2,[widthlength,widthlength]);
    I3 = Image(segment_y+1:2*segment_y, 1:segment_x);
    I3 = imresize(I3,[widthlength,widthlength]);
    I4 = Image(segment_y+1:2*segment_y, segment_x+1:2*segment_x);
    I4 = imresize(I4,[widthlength,widthlength]);
    
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    imwrite(I1,filename);
    w=w+1;
    
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    imwrite(I2,filename);
    w=w+1;
    
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    imwrite(I3,filename);
    w=w+1;
    
    if(k+w+K_p)<10
        filename = strcat(N,'\','0000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=10 && k+w+K_p<=99) 
        filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=100 && k+w+K_p<=999)
        filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
    elseif(k+w+K_p>=1000 && k+w+K_p<=9999)
        filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
    else
        filename = strcat(N,'\',num2str(k+w+K_p),'.png');
    end
    imwrite(I4,filename);
end
%% Writing 
% if(k+w+K_p)<10
%     filename = strcat(N,'\','000',num2str(k+w+K_p),'.png');
% elseif(k+w+K_p>=10 && k+w+K_p<=99) 
% 	filename = strcat(N,'\','00',num2str(k+w+K_p),'.png');
% elseif(k+w+K_p>=100 && k+w+K_p<=999)
%     filename = strcat(N,'\','0',num2str(k+w+K_p),'.png');
% else
%     filename = strcat(N,'\',num2str(k+w+K_p),'.png');
% end
% Image = imresize(Image, [widthlength widthlength]);
% imwrite(Image,filename);
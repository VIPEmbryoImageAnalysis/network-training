% Only need to edit line 2 
file = '/Users/wilzh/Desktop/Fall 2019/Videos/mouse4981.avi';
 [filepath,name,ext] = fileparts(file);
 mkdir(name);
 vidObj = VideoReader(file);
 time_to_remember = vidObj.CurrentTime;
 numFrames = 0;
 path = strcat(filepath,'/',name);
 
    while hasFrame(vidObj)
        readFrame(vidObj);
        numFrames = numFrames + 1;
    end
    
    vidObj.CurrentTime = time_to_remember;
    
    for k=1:numFrames
        Image = readFrame(vidObj);
        if(k<10)
            filename=strcat(name,'/','00',num2str(k),'_',name,'.png');
        elseif(k>=10 && k<=99)
            filename=strcat(name,'/','0',num2str(k),'_',name,'.png');    
        else
            filename=strcat(name,'/',num2str(k),'_',name,'.png');
        end
        destination = strcat(filepath,'/',name,'/',filename);

        imwrite(Image,filename,'png');
    end
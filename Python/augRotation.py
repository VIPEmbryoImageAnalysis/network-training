#Rotation Augmentation Script

from PIL import Image
import os
import glob

#Define path to original images
org_dir = "/Users/christinembramos/Desktop/Embryo Image Analysis/ExtractedFrames"
ext = ".png"

#Create new folder
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/RotatedFrames')

#Define degrees to rotate images
deg = [90, 180, 270]

#Loop to rotate images with .png extension
for file_name in glob.iglob(os.path.join(org_dir, "*" + ext)):
    file_nameNew = os.path.basename(file_name)
    img = Image.open(file_name)
    for degrees in deg:
        #Rename augmented images to have angle at the end of the file name
        rotateName = "_r{:03d}{:s}".format(degrees, ext)
        file_nameRotated = file_nameNew.replace(ext, rotateName)
        #Rotate images
        rotatedImg = img.rotate(degrees)
        #Save images to specified folder
        save_dir = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/RotatedFrames'
        file_path = os.path.join(save_dir, file_nameRotated)
        rotatedImg.save(file_path)
        #Print statement after images are augmented
        print("ROTATED: {:s} by {:3d} degrees to {:s}".format(file_nameNew, degrees, save_dir))

print("\nTask Completed.")
    

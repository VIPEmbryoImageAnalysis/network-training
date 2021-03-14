#Mirror Augmentation Script

from PIL import Image
import os
import glob

#Define path to original images
org_dir = "/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER"
ext = ".png"

#Create new folder
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/MirroredFrames')

#Run loop
for file_name in glob.iglob(os.path.join(org_dir, "*" + ext)):
    file_nameNew = os.path.basename(file_name)
    img = Image.open(file_name)
        
    #Mirror images over Y axis
    imgY = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    #Rename augmented images to have mY at the end of the file name
    mirrorName_Y = "_mY{:s}".format(ext)
    file_nameY = file_nameNew.replace(ext, mirrorName_Y)
        
    #Mirror images over X axis
    imgX = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    #Rename augmented images to have mX at the end of the file name
    mirrorName_X = "_mX{:s}".format(ext)
    file_nameX = file_nameNew.replace(ext, mirrorName_X)
        
    #Save images to specified folder
    save_dir = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/MirroredFrames'
    file_pathY = os.path.join(save_dir, file_nameY)
    file_pathX = os.path.join(save_dir, file_nameX)
    imgY.save(file_pathY)
    imgX.save(file_pathX)
        
    #Print statement after images are augmented
    print("MIRRORED: {:s} HORIZONTALLY to {:s}".format(file_nameNew, save_dir))
    print("MIRRORED: {:s} VERTICALLY to {:s}".format(file_nameNew, save_dir))

print("\nTask Completed.")

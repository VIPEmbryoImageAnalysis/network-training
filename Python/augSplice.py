#Splice Augmentation Script

from PIL import Image
import os
import glob
import image_slicer

#Define path to original images
org_dir = "/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER"
ext = ".png"

#Create new folder
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/SplicedFrames')

#Run loop
for file_name in glob.iglob(os.path.join(org_dir, "*" + ext)):
    file_nameNew = os.path.basename(file_name)
    img = Image.open(file_name)

    #Split image into 4 equal pieces
    pieces = image_slicer.slice(file_name, 4, save=False)
        
    #Save images to specified folder
    save_dir = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/SplicedFrames'
    image_slicer.save_tiles(pieces, directory=save_dir, prefix=file_name, format='png')
        
    #Print statement after images are augmented
    print("Split {:s} into 4 pieces in {:s}".format(file_nameNew, save_dir))

print("\nTask Completed.")

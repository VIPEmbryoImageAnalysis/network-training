#Performs all augmentation methods:
    #Mirror, Rotate, Splice

from PIL import Image
import os
import glob
import image_slicer

#Define path to original images
org_dir = "/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER"
ext = ".png"

#Create new folder for each augmentation method
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Mirrored')
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Rotated')
os.makedirs('/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Spliced')

#Loop for all files with .png extension
for file_name in glob.iglob(os.path.join(org_dir, "*" + ext)):
    file_nameNew = os.path.basename(file_name)
    img = Image.open(file_name)

    #Mirror images over Y axis
    imgY = img.transpose(Image.FLIP_LEFT_RIGHT)

    #Mirror images over X axis
    imgX = img.transpose(Image.FLIP_TOP_BOTTOM)

    #Rename augmented images to have mY at the end of the file name
    mirrorName_Y = "_mY{:s}".format(ext)
    file_nameY = file_nameNew.replace(ext, mirrorName_Y)
      
    #Rename augmented images to have mX at the end of the file name
    mirrorName_X = "_mX{:s}".format(ext)
    file_nameX = file_nameNew.replace(ext, mirrorName_X)

    #Save images to specified folder
    save_dirMirror = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Mirrored'
    file_pathY = os.path.join(save_dirMirror, file_nameY)
    file_pathX = os.path.join(save_dirMirror, file_nameX)
    imgY.save(file_pathY)
    imgX.save(file_pathX)

    #Print statements after images are augmented
    print("MIRRORED: {:s} HORIZONTALLY to {:s}".format(file_nameNew, save_dirMirror))
    print("MIRRORED: {:s} VERTICALLY to {:s}".format(file_nameNew, save_dirMirror))

    #Define degrees for rotations
    deg = [90, 180, 270]

    for degrees in deg:   
        #Rotate images
        rotatedImg = img.rotate(degrees)
    
        #Rename augmented images to have angle at the end of the file name
        rotateName = "_r{:03d}{:s}".format(degrees, ext)
        file_nameRotated = file_nameNew.replace(ext, rotateName)

        #Save images to specified folder
        save_dirRotate = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Rotated'
        file_pathRotate = os.path.join(save_dirRotate, file_nameRotated)
        rotatedImg.save(file_pathRotate)
        
        print("ROTATED: {:s} by {:3d} degrees to {:s}".format(file_nameNew, degrees, save_dirRotate)) 

    #Split image into 4 equal pieces
    pieces = image_slicer.slice(file_name, 4, save=False) [0]
    
    #Save images to specified folder
    save_dirSplice = '/Users/christinembramos/Desktop/Embryo Image Analysis/Augmented/Spliced'
    file_pathSplice = os.path.join(save_dirSplice, file_nameNew)
    pieces.save(file_pathSplice)
        
    print("Split {:s} into 4 pieces in {:s}".format(file_nameNew, save_dirSplice))
    
print("Task Completed.\n")

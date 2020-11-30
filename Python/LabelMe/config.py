# Configurations for training a network

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (256, 256, 3)

# set your classification mode (binary or multi)
mode = 'multi'

# model_name (unet)
model_name = 'unet_'+mode

# log dir for tensorboard
logbase = 'logs'

# classes are defined in color hues
# background and embryo
hues = {'Embryo': 240
        }

labels = sorted(hues.keys())

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1

assert imshape[0]%32 == 0 and imshape[1]%32 == 0,\
    "imshape should be multiples of 32. comment out to test different imshapes."

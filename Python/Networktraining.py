from train_MATLAB_data.models.unet import unet
import tensorflow as tf
import matplotlib.pyplot as plt

# lines 6-9 required to use GPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load model
model = unet(n_classes=3 , input_height=480, input_width=480 )

# train model with dataset
model.train(
     train_images = "1790Dataset/EImages/",
     train_annotations = "1790Dataset/ELabeled/",
     checkpoints_path = "1790Dataset/Checkpoints/", epochs=30
)

# evaluating the model 
print(model.evaluate_segmentation(
    inp_images_dir="TestingSetHCwells/Images" ,annotations_dir="TestingSetHCwells/Labels",))

# apply test segmentation
model.predict_multiple(inp_dir="tester/Images",
                       out_dir="tester/PythonSeg", overlay_img=True)



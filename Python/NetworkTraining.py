# Network Training Script - IN PROGRESS

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from model import unet

# Paths for images and labels
train_images = '/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER'  # Path used only as placeholder
train_labels = '/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER'  # Path used only as placeholder

test_images = '/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER'   # Path used only as placeholder
test_labels = '/Users/christinembramos/Desktop/Embryo Image Analysis/TESTFOLDER'   # Path used only as placeholder

# Call model
model = unet(pretrained=False, base=2)

# Feed the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Plot predictions
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

    





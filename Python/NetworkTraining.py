# Network Training Script

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import model from model.py
from model.py import model

# Paths for images and labels
train_images = 
train_labels = 

test_images =
test_labels =

# Feed the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

    





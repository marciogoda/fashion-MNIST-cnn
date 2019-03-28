from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

print("tain_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)
plt.imshow(train_images[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

print("image:", train_images[0])


model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, input_shape=(28,28,1)),

    keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

model.fit(train_images, train_labels, batch_size=64, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)


print('Test accuracy:', test_acc)

model.save('mnist.h5')



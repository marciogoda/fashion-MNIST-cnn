from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0.08,
        zoom_range = 0.08, 
        width_shift_range=0.08,  
        height_shift_range=0.08,  
        horizontal_flip=True)  


datagen.fit(train_images)

log_dir = "./logs/"

tb_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)

print("tain_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)

model.fit_generator(datagen.flow(train_images, train_labels, batch_size=512), 
                                    validation_data=(test_images,test_labels), 
                                    epochs=50, steps_per_epoch=train_images.shape[0]//512,
                                    validation_steps=test_images.shape[0]//512,
                                    use_multiprocessing=True,
                                    callbacks=[tb_callback])


test_loss, test_acc = model.evaluate(test_images, test_labels, callbacks=[tb_callback])


print('Test accuracy:', test_acc)

model.save('mnist.h5')





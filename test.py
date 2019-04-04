from __future__ import absolute_import, division, print_function

import sys
import requests
import tensorflow as tf
import numpy as np
from tensorflow import keras
from io import BytesIO
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt


resp = requests.get(sys.argv[1])

img = Image.open(BytesIO(resp.content))

gray_img = img.convert('L')

tn_image = gray_img.resize((28,28))

inverted_image = PIL.ImageOps.invert(tn_image)

imarr2 = np.array(inverted_image)

imarr3 = imarr2 / 255.0

model = keras.models.load_model('mnist.h5')

test_image = imarr2.reshape(1,28,28,1)

predictions = model.predict(test_image)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("Prediction:", class_names[np.argmax(predictions[0])], predictions[0][np.argmax(predictions[0])])



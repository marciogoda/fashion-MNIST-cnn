from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps

import sys
import requests
import tensorflow as tf
import numpy as np
import urllib.parse

from tensorflow import keras
from io import BytesIO
from PIL import Image
import PIL.ImageOps

app = Flask(__name__)
api = Api(app)
model = keras.models.load_model('./models/mnist.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_image(url):
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content))
    gray_img = img.convert('L')
    tn_image = gray_img.resize((28,28))
    inverted_image = PIL.ImageOps.invert(tn_image)
    imarr2 = np.array(inverted_image)
    imarr3 = imarr2 / 255.0
    return imarr3.reshape(1,28,28,1)



class Predictions (Resource):
    def get(self):
        url = request.args.get('url')
        decoded_url = urllib.parse.unquote(url)
        image = get_image(decoded_url)
        predictions = model.predict(image)
        return {'prediction':{'class': class_names[np.argmax(predictions[0])], 
                'confidence': str(predictions[0][np.argmax(predictions[0])])}}


api.add_resource(Predictions, '/predict')

if __name__ == '__main__':
    app.run(port='8080')
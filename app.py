from __future__ import division, print_function
# coding: utf-8
import sys
import os
import re
import glob
import numpy as np
from PIL import Image as pil_image

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Flask Utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)


Model = load_model('mobileNet_new_model.h5')


lesion_classes_dict = {
    0 : 'Melanocytic nevi',
    1 : 'Melanoma',
    2 : 'Benign keratosis-like lesions ',
    3 : 'Basal cell carcinoma',
    4 : 'Actinic keratoses',
    5 : 'Vascular lesions',
    6 : 'Dermatofibroma'
}

# def model_predict(img_path, Model):
#     img = image.load_img(img_path, target_size=(224, 224, 3))
    
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
    
    
#     preds = Model.predict(x)
#     return preds

def model_predict(img_path, Model):
    # Load the image with the correct target size (height, width)
    img = load_img(img_path, target_size=(224, 224))
    
    # Convert the image to an array
    x = img_to_array(img)
    
    # Expand dimensions to match the expected input shape for the model
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image if necessary (e.g., scaling pixel values)
    x = x / 255.0  # Example normalization if your model expects pixel values in [0, 1]

    # Make predictions using the model
    preds = Model.predict(x)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main Page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST']) # type: ignore
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to './uploads'
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,
                                'uploads',
                                secure_filename(f.filename) # type: ignore
                                ) # type: ignore
        f.save(file_path)

        # Make Prediction
        preds = model_predict(file_path, Model)

        # Process your results for Humans

        pred_class = preds.argmax(axis = -1)
        pr = lesion_classes_dict[pred_class[0]]
        result = str(pr)
        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
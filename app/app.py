import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


IMAGE_WIDTH = 128
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_DEPTH = 3


def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction 
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0]


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)
    

model = load_model('models/manufacturer.h5')
model.summary()

st.title("Identification d'avion")

uploaded_file = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)

predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
if predict_btn:
    prediction = predict_image(uploaded_file, model)
    st.write(f"C'est un : {prediction}")
    # Exemple si les f-strings ne sont pas dispo.
    #st.write("C'est un : {}".format(prediction)

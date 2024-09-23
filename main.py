import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

model = joblib.load('C:\your_trained_model.pkl')
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    return img_array

def predict_image(img):
    img_flattened = img.reshape(1, -1)
    prediction = model.predict(img_flattened)
    return 'Cat' if prediction[0] == 0 else 'Dog'
st.title("Cat and Dog Image Classifier")
st.write("Upload a cat or dog image to see the prediction.")
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    predicted_class = predict_image(image)
    st.write(f"The predicted class is: **{predicted_class}**")
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)


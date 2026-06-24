import streamlit as st
import pickle
import numpy as np
from PIL import Image

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Resize to model input size
    image = image.resize((64, 64))

    # Convert to array
    img_array = np.array(image).flatten().reshape(1, -1)

    prediction = model.predict(img_array)

    label = "Cat" if prediction[0] == 0 else "Dog"

    st.success(f"Prediction: {label}")

"""
Chest X-ray Classifier (Normal vs Pneumonia) - Streamlit app

Upload an X-ray image -> convert to RGB -> resize to 128x128 -> scale to 0-1
-> CNN model (xray.keras) -> Normal / Pneumonia
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128          # same size the model was trained with
MODEL_PATH = "xray.keras"

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩻")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image):
    """Same steps as the training notebook: RGB -> resize -> /255 -> add batch dimension."""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(img_array, axis=0)


st.title("🩻 Chest X-ray Classifier")
st.write("Upload a chest X-ray image and the model will predict whether it looks **Normal** or shows **Pneumonia**.")
st.caption("Educational project only. This is not a medical diagnosis tool.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to get a prediction.")
else:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("This file could not be opened as an image. Please upload a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model = load_model()
    except Exception as error:
        st.error(f"Model file '{MODEL_PATH}' could not be loaded. Make sure it is in the same folder as app.py.")
        st.exception(error)
        st.stop()

    if st.button("Predict"):
        with st.spinner("Analysing image..."):
            prob = float(model.predict(preprocess(image), verbose=0)[0][0])

        # The model outputs the probability of class 1 = PNEUMONIA (sigmoid output)
        if prob > 0.5:
            st.error(f"Prediction: **Pneumonia**  ({prob * 100:.1f}% confidence)")
        else:
            st.success(f"Prediction: **Normal**  ({(1 - prob) * 100:.1f}% confidence)")

        st.write("Class probabilities")
        st.progress(prob, text=f"Pneumonia: {prob * 100:.1f}%")
        st.progress(1 - prob, text=f"Normal: {(1 - prob) * 100:.1f}%")

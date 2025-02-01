import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image


file_id = "1nB_He748NOEOJvwo6Q6gnCCcRoDmtLN4"
url = 'https://drive.google.com/file/d/1nB_He748NOEOJvwo6Q6gnCCcRoDmtLN4/view?usp=sharing'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)


# Model path
model_path = "trained_plant_disease_model.keras"


def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  


st.set_page_config(
    page_title="Leaf Disease Detection for Smart Farming",
    layout="centered"
)

# Background style
page_bg_style = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #EAF4D3;
}
[data-testid="stSidebar"] {
    background-color: #D1E7B8;
}
h1, h2 {
    color: #2E8B57;
    font-family: 'Arial Black', sans-serif;
}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

# Sidebar with navigation
st.sidebar.title(" Leaf Disease Detection for Smart Farming")
app_mode = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Disease Recognition"]
)

# Header image with adjusted width
header_img = Image.open("Diseases.jpg")
st.image(header_img, caption="Leaf Disease Detection for Smart Farming", use_column_width=False, width=600)

# Home page
if app_mode == "Home":
    st.markdown(
        """
        <h1 style='text-align: center;'>Welcome to the Leaf Disease Detection App </h1>
        <p style='text-align: center; color: #556B2F;'>This tool helps farmers detect potato leaf diseases using AI technology.</p>
        """,
        unsafe_allow_html=True
    )
    st.info("Navigate to the 'Disease Recognition' tab to upload an image for prediction.")

# Prediction page
elif app_mode == "Disease Recognition":
    st.markdown(
        "<h2> Upload and Analyze Your Leaf Image</h2>",
        unsafe_allow_html=True
    )
    
    # File uploader
    test_image = st.file_uploader("Choose a leaf image file:", type=["jpg", "jpeg", "png"])
    
    # Display image preview with adjusted width
    if test_image:
        img = Image.open(test_image)
        st.image(img, caption="Uploaded Image", use_column_width=False, width=700)
        
        # Predict button
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(test_image)
                class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                st.snow()
                st.success(f"Model Prediction: **{class_names[result_index]}**")

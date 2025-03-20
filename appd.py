import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# Load the saved model
model = joblib.load('my_1model.pkl')

# Set page title and description
st.set_page_config(page_title="Medical Image Classifier", layout="wide")
st.title("🩺 Chest X-ray Classification")
st.write("""
This app predicts the class of chest X-ray images into one of three categories:
- COVID-19
- Lung Opacity
- Viral Pneumonia

Upload a chest X-ray image (JPEG/PNG) for analysis.
""")

# Create file uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# Class names mapping
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Viral_Pneumonia']


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array


def predict_class(image_array):
    """Make prediction using the loaded model"""
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    return CLASS_NAMES[predicted_index], confidence, prediction[0]


if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # Make and display prediction
    with col2:
        st.subheader("Analysis Results")
        with st.spinner('Processing image...'):
            try:
                # Preprocess and predict
                img_array = preprocess_image(uploaded_file)
                class_name, confidence, probabilities = predict_class(img_array)

                # Display results
                st.success("Prediction Complete!")
                st.markdown(f"**Predicted Class:** {class_name}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")

                # Show probability distribution
                st.subheader("Probability Distribution")
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                    progress = int(prob * 100)
                    st.markdown(f"{class_name}:")
                    st.progress(progress)
                    st.write(f"{prob * 100:.2f}%")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

# Add some footer information
st.markdown("---")
st.markdown(
    "*This tool is for research purposes only. Always consult a healthcare professional for medical diagnosis.*")
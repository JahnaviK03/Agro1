import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os

# Load image
img = Image.open("st.image("https://example.com/fertilizer.png", use_column_width=True)")  # Use an appropriate image
st.image(img, use_column_width=True)

# Load the model
model = pickle.load(open('fertilizer_model.pkl', 'rb'))

# Mapping for soil and crop types to numeric values (must match training encoding)
soil_mapping = {'Clayey': 0, 'Loamy': 1, 'Sandy': 2}
crop_mapping = {'Rice': 0, 'Sugarcane': 1, 'Maize': 2, 'Cotton': 3, 'Wheat': 4, 'Barley': 5}  # adjust as needed

# Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center;'>SMART FERTILIZER RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    st.sidebar.title("AgriSens")
    st.sidebar.header("Enter Soil and Crop Details")

    temperature = st.sidebar.number_input("Temperature (in Celsius)", 0, 60, 25)
    humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 50)
    nitrogen = st.sidebar.number_input("Nitrogen Content in Soil (ppm)", 0, 200, 100)
    potassium = st.sidebar.number_input("Potassium Content in Soil (ppm)", 0, 200, 100)
    phosphorus = st.sidebar.number_input("Phosphorous Content in Soil (ppm)", 0, 200, 100)

    soil_type = st.sidebar.selectbox("Select Soil Type", list(soil_mapping.keys()))
    crop_type = st.sidebar.selectbox("Select Crop Type", list(crop_mapping.keys()))

    if st.sidebar.button("Predict"):
        try:
            features = np.array([[temperature, humidity, nitrogen, potassium, phosphorus,
                                  soil_mapping[soil_type], crop_mapping[crop_type]]])
            prediction = model.predict(features)
            st.success(f"Recommended Fertilizer: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()

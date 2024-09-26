import os
import streamlit as st
from pathlib import Path
import logging

import tempfile
import cv2
import numpy as np

from sample_utils.download import download_file

# Model
from ultralytics import YOLO

from PIL import Image
from io import BytesIO

# Extraction of coordinates from image
import piexif

# Map display
import folium
from streamlit_folium import folium_static

# <<< Code starts here >>>
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Balloon Detection App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# https://github.com/hamdani2020/balloon/blob/main/balloon.pt

# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_URL = "https://github.com/hamdani2020/balloon/raw/main/balloon.pt"
MODEL_LOCAL_PATH = ROOT / "models" / "balloon.pt"

# Ensure the models directory exists
os.makedirs(ROOT / "models", exist_ok=True)

# Download file with logging
def download_file_with_logging(url, local_path, expected_size):
    logger.info(f"Downloading file from {url} to {local_path}")
    download_file(url, local_path, expected_size)
    logger.info(f"Download complete. File saved to {local_path}")

# Load model with caching
@st.cache_resource()
def load_model():
    logger.info("Entering load_model function")
    if not MODEL_LOCAL_PATH.exists():
        logger.info(f"Model not found at {MODEL_LOCAL_PATH}. Downloading...")
        download_file_with_logging(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)
    else:
        logger.info(f"Model found at {MODEL_LOCAL_PATH}")
    logger.info("Loading YOLO model")
    model = YOLO(MODEL_LOCAL_PATH)
    logger.info("YOLO model loaded successfully")
    return model

# Load the model
logger.info("About to load the model")
net = load_model()
logger.info("Model loading complete")

# Balloon is the only class
CLASSES = ["Balloon"]

# Title and introduction
title = """<h1>Balloon Detection App</h1>"""
st.markdown(title, unsafe_allow_html=True)
subtitle = """
Upload an image to detect balloons and view their location on a map
"""
st.markdown(subtitle)

# File upload section
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Extract GPS info from image
def extract_gps_info(image):
    try:
        exif_dict = piexif.load(image.info["exif"])
        gps_info = exif_dict.get("GPS", {})
        
        if gps_info:
            lat = gps_info.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
            lon = gps_info.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)
            
            if lat and lon and lat_ref and lon_ref:
                lat = sum(float(x) / float(y) for x, y in lat) * (-1 if lat_ref == b'S' else 1)
                lon = sum(float(x) / float(y) for x, y in lon) * (-1 if lon_ref == b'W' else 1)
                return lat, lon
    except Exception as e:
        logger.error(f"Error extracting GPS info: {e}")
    
    return None, None

# Process image and detect balloons
def process_image(image_file):
    logger.info(f"Processing image: {image_file.name}")
    image = Image.open(image_file)
    lat, lon = extract_gps_info(image)
    _image = np.array(image)
    
    # Resize image for better processing
    h_ori, w_ori = _image.shape[:2]
    image_resized = cv2.resize(_image, (720, 640), interpolation=cv2.INTER_AREA)
    
    # Run prediction
    results = net.predict(image_resized, conf=score_threshold)
    
    # Annotate the image with detection results
    annotated_frame = results[0].plot()
    image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
    
    return _image, image_pred, lat, lon

# Create map with detected balloons
def create_map(coordinates):
    m = folium.Map(location=[0, 0], zoom_start=2)
    for lat, lon in coordinates:
        if lat is not None and lon is not None:
            folium.Marker([lat, lon], popup="Balloon detected").add_to(m)
    return m

# Process images and display results
if uploaded_files:
    all_coordinates = []
    for image_file in uploaded_files:
        st.write(f"### Processing: {image_file.name}")
        # Process and display images
        original_image, predicted_image, lat, lon = process_image(image_file)
        all_coordinates.append((lat, lon))
        
        col1, col2 = st.columns(2)
        # Display original image
        with col1:
            st.write("#### Original Image")
            st.image(original_image)
        # Display predicted image
        with col2:
            st.write("#### Predictions")
            st.image(predicted_image)
            
        # Option to download the predicted image
        buffer = BytesIO()
        download_image = Image.fromarray(predicted_image)
        download_image.save(buffer, format="PNG")
        download_image_byte = buffer.getvalue()
        st.download_button(
            label="Download Predicted Image",
            data=download_image_byte,
            file_name=f"Predicted_{image_file.name}",
            mime="image/png"
        )
        
        if lat is not None and lon is not None:
            st.write(f"GPS Coordinates: Latitude {lat:.6f}, Longitude {lon:.6f}")
        else:
            st.write("No GPS coordinates found in the image metadata.")
    
    # Create and display the map
    st.write("### Map of Detected Balloons")
    map = create_map(all_coordinates)
    folium_static(map)

logger.info("All images processed successfully")
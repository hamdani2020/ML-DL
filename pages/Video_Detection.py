import os
import streamlit as st
from pathlib import Path
import logging
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from sample_utils.download import download_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Balloon Video Detection App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
@st.cache_resource
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
st.title("Balloon Video Detection App")
st.markdown("Upload a video to detect balloons")

# File upload section
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Function to process video
def process_video(video_file):
    logger.info(f"Processing video: {video_file.name}")
    
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.flush()

    try:
        # Open the video file
        video = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        # Create a temporary file for the output video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
        
        # Process the video
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Run prediction on the frame
            results = net.predict(frame, conf=score_threshold)
            
            # Annotate the frame with detection results
            annotated_frame = results[0].plot()
            
            # Write the frame to the output video
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        # Release everything when job is finished
        video.release()
        out.release()
        
        logger.info("Video processing complete")
        return output_file.name
    
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise e

    finally:
        # Clean up the temporary input file
        os.remove(tfile.name)
        logger.info(f"Temporary video file {tfile.name} removed.")

# Store the processed video path in session state
if 'processed_video_path' not in st.session_state:
    st.session_state['processed_video_path'] = None

if uploaded_file is not None and st.session_state['processed_video_path'] is None:
    st.write("Processing video... This may take a while.")
    processed_video_path = process_video(uploaded_file)
    st.session_state['processed_video_path'] = processed_video_path
    
# Show the processed video if it exists in session state
if st.session_state['processed_video_path'] is not None:
    st.write("### Processed Video")
    st.video(st.session_state['processed_video_path'])
    
    # Option to download the processed video
    with open(st.session_state['processed_video_path'], "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up the processed video file after download
    if st.button('Clear Processed Video'):
        # Safely remove the processed video from session state and delete the file
        os.remove(st.session_state['processed_video_path'])
        st.session_state['processed_video_path'] = None

        # Reset by updating the query parameters to trigger a page rerun
        st.experimental_set_query_params()  # This will cause Streamlit to rerun the app without any additional steps


logger.info("Video processing script completed")

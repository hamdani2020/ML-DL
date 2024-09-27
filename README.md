# Balloon Detection App 🎈

## Project Description
The **Balloon Detection App** is a machine learning application designed to detect and locate balloons in images and video streams using advanced computer vision techniques. Built with **Streamlit**, the app utilizes the YOLO (You Only Look Once) object detection model to provide real-time and high-accuracy balloon detection. Additionally, it extracts GPS metadata from uploaded images (if available) and displays the balloon locations on an interactive map.

This project demonstrates how computer vision can be integrated into a user-friendly web-based interface, with support for both static images and video streams.

## Features
- **Real-time Balloon Detection**: Detect balloons in real-time from your webcam or other video streams.
- **Video Detection**: Upload video files for balloon detection.
- **Image Detection**: Upload images for detection, and the app will highlight balloons.
- **GPS Metadata Extraction**: If an image contains GPS data, the app extracts and displays latitude and longitude.
- **Interactive Map**: Detected balloon locations are visualized on a **Folium** map.
- **Downloadable Results**: Download processed images or video frames with detected balloons.

## How to Install

### Step 1: Clone the Repository
First, clone the project repository from GitHub:
```bash
git clone https://github.com/hamdani2020/ML-DL.git
cd ML-DL
```

### Step 2: Create a Virtual Environment
To avoid conflicts with global packages, it's recommended to create a virtual environment:
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Create a `requirements.txt` File
Ensure the `requirements.txt` file contains the necessary dependencies:
```
opencv-python
numpy
ultralytics
pillow
piexif
folium
streamlit
streamlit-folium
streamlit-webrtc
av
```

### Step 5: Install the Dependencies
Install the necessary Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 6: Download the YOLO Model Weights
The app requires the YOLO model for balloon detection. The pre-trained model weights will be downloaded automatically when the app is first run, or you can manually download them from:
- [Balloon Model Weights](https://github.com/hamdani2020/balloon/raw/main/balloon.pt)

Save the weights file to the `models` folder in your project directory.

## App Usage

### Step 1: Run the App
Once the installation is complete, you can launch the Streamlit app using:
```bash
streamlit run app.py
```

### Step 2: Interact with the App
Open your web browser and navigate to `http://localhost:8501` to use the app. You can:
- **Real-Time Detection**: Use your webcam or other video sources for real-time balloon detection.
- **Video Detection**: Upload `.mp4` or other supported video formats for balloon detection.
- **Image Detection**: Upload `.png`, `.jpg`, `.jpeg`, or `.webp` images for balloon detection.
- **Set Confidence Threshold**: Adjust the confidence threshold to control the sensitivity of the balloon detection model.

### Step 3: View Results
- **Balloon Detection**: The app will display original images or video frames, and show where balloons have been detected.
- **Download Processed Files**: You can download processed images or video frames with detection results.
- **Map Display**: If GPS metadata is available in an image, the app will show the balloon's location on an interactive map.

## Project Structure
```
balloon-detection-app/
│
├── images/                       # Directory for storing images  
├── models/                       # Directory for storing YOLO model weights
├── models/balloon.pt             # Balloon model weights
├── pages/                        # Directory for storing pages for streamlit
├── pages/about.py                # About page
├── pages/image_detection.py      # Image detection page
├── pages/video_detection.py      # Video detection page
├── pages/real_time_detection.py  # Real-time detection page
├── sample_utils/                 # Directory for download utils
├── app.py                        # Main application file
├── requirements.txt              # File containing Python dependencies
└── README.md                     # Project documentation
```

## Future Features (Planned)
- **Improved Video Processing**: Enhance performance for processing longer video streams.
- **Custom Object Detection**: Allow users to upload their own trained models for detecting custom objects beyond balloons.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request if you have suggestions, improvements, or bug fixes. For major changes, please open an issue first to discuss the changes.

## License
This project is licensed under the MIT License.
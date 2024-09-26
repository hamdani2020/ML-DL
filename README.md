# Balloon Detection App 🎈

## Project Description
The **Balloon Detection App** is designed to detect and locate balloons in images using advanced computer vision techniques. Built with **Streamlit**, this app employs the YOLO (You Only Look Once) object detection model to provide accurate and real-time balloon detection. Additionally, it extracts GPS metadata from the uploaded images (if available) and displays the balloon locations on an interactive map.

This project demonstrates the power of computer vision for object detection in a user-friendly, web-based interface.

## Features
- **Real-time Balloon Detection**: Upload images, and the app will detect and highlight balloons.
- **GPS Metadata Extraction**: If an image contains GPS data, the app will extract and display the latitude and longitude.
- **Interactive Map**: Detected balloon locations are shown on a **Folium** map within the app.
- **Downloadable Results**: Processed images with detections can be downloaded.

## How to Install

### Step 1: Clone the Repository
First, clone the project repository from GitHub:
```bash
git clone git clone https://github.com/hamdani2020/ML-DL.git
cd ML-DL
```

### Step 2: Create a Virtual Environment
To avoid conflicts with global packages, it's best to create a virtual environment:
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
streamlit
opencv-python
numpy
ultralytics
pillow
piexif
folium
streamlit-folium
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
Open your web browser and go to `http://localhost:8501` to use the app. You can:
- **Upload Images**: Upload `.png`, `.jpg`, `.jpeg`, or `.webp` images for balloon detection.
- **Set Confidence Threshold**: Adjust the confidence threshold for detection to control the sensitivity of the balloon detection model.

### Step 3: View Results
- **Balloon Detection**: The app will display the original image and the image with detected balloons highlighted.
- **Download Processed Images**: You can download the image with the detection results.
- **Map Display**: If GPS metadata is available in the image, the app will show the location of the balloons on an interactive map.

## Project Structure
```
balloon-detection-app/
│
├── images/                 # Directory for storing images  
├── pages/                  # Directory for storing pages for streamlit
├── app.py                  # Main application file
├── models/                 # Directory for storing YOLO model weights
├── requirements.txt        # File containing Python dependencies
└── README.md               # Project documentation
```

## Future Features (Planned)
- **Video Stream Detection**: Extend the app to handle live video streams for balloon detection.
- **Real-time Detection**: Implement real-time detection for video streams.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request if you have suggestions, improvements, or bug fixes. For major changes, please open an issue first to discuss the changes.

## License
This project is licensed under the MIT License.
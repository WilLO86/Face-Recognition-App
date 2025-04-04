# Face Recognition App with Streamlit

## Project Description
This application is a Face Recognition App built using Streamlit and various computer vision libraries. The app performs the following tasks:
- **Face Detection:** Detects faces within an image using an OpenCV DNN model.
- **Face Recognition:** Compares detected faces with known faces to identify them.
- **Smile Detection:** Identifies smiles in the image using a Haar Cascade classifier.

## Implementation
The app is implemented using the following technologies:
- **Streamlit:** For building an interactive web interface.
- **OpenCV:** For image processing and face detection using a deep neural network (DNN) model.
- **face_recognition:** For encoding and comparing facial features.
- **Pillow (PIL):** For image manipulation and conversion.
- **NumPy:** For handling numerical operations on image arrays.
- **Pandas:** For displaying detection details in a tabular format.

### Code Logic Overview
1. **Model Loading:**  
   The application loads a pre-trained face detection model (`res10_300x300_ssd_iter_140000_fp16.caffemodel`) along with its configuration (`deploy.prototxt`) using OpenCV's DNN module.
2. **Face Detection:**  
   It processes the input image (either uploaded or captured via webcam) to detect faces and draws bounding boxes around them.
3. **Face Recognition:**  
   In Face Recognition mode, the app compares detected faces against a set of known face images provided by the user.
4. **Smile Detection:**  
   Alternatively, the app can detect smiles using a Haar Cascade classifier.
5. **User Interface:**  
   An intuitive sidebar provides controls for selecting the operation mode, adjusting the confidence threshold, and choosing the image source (upload or camera).
6. **Output:**  
   The app displays the original and processed images side-by-side, shows detection details in a table, and offers a download link for the processed image.

## Enhancements and Customizations
This version of the application includes the following improvements over the original template:
- **Webcam Integration:** Users can capture images directly from their webcam.
- **Interactive Sidebar:** A sidebar with controls for selecting the mode, adjusting the confidence threshold, and choosing the image source.
- **Download Option:** A feature to download the processed image directly from the app.
- **Detection Details:** An interactive table displaying bounding box coordinates and confidence scores for detected faces.

## Usage Instructions

### Running Locally
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd Face-Recognition-App

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition  # Make sure to install this library: pip install face_recognition
import time
import base64
from io import BytesIO
import pandas as pd

# Function to generate a download link for the processed image.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Load the face detection model using OpenCV DNN.
@st.cache_resource()
def load_detection_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Function to detect faces using the DNN model.
def detect_faces_dnn(net, image, conf_threshold=0.5):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    h, w = image.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append((x1, y1, x2, y2, confidence))
    return bboxes

# Function to recognize faces using the face_recognition library.
def recognize_faces(known_encodings, known_names, image, bboxes):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # For each bounding box, obtain the face encoding.
    encodings = face_recognition.face_encodings(rgb_image, [(y1, x2, y2, x1) for (x1, y1, x2, y2, conf) in bboxes])
    names = []
    for encoding in encodings:
        # Compare the detected face encoding with the known encodings.
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = "Unknown"
        names.append(name)
    return names

# Function to load the Haar cascade classifier for smile detection.
@st.cache_resource()
def load_smile_cascade():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return cascade

# Function to detect smiles in the image.
def detect_smiles(image, scaleFactor=1.7, minNeighbors=22):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smile_cascade = load_smile_cascade()
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return smiles

# Main function of the application.
def main():
    st.set_page_config(page_title="Face Recognition App", layout="wide")
    st.title("Face Recognition App with Streamlit and OpenCV")
    
    # Sidebar for selecting mode and parameters.
    mode = st.sidebar.selectbox("Select Operation Mode", 
                                ["Face Detection", "Face Recognition", "Smile Detection"])
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    use_source = st.sidebar.radio("Image Source", ["Upload Image", "Use Camera"])
    
    # For Face Recognition mode, load known faces.
    if mode == "Face Recognition":
        st.sidebar.markdown("### Load Known Faces")
        known_files = st.sidebar.file_uploader("Upload known face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        known_encodings = []
        known_names = []
        if known_files:
            for file in known_files:
                img_known = face_recognition.load_image_file(file)
                encodings = face_recognition.face_encodings(img_known)
                if encodings:
                    known_encodings.append(encodings[0])
                    # Use the file name (without extension) as the label.
                    known_names.append(file.name.split('.')[0])
            st.sidebar.success("Known faces loaded successfully")
    
    # Select the input image, either by upload or using the camera.
    if use_source == "Upload Image":
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Take a Photo")
    
    if image_file is not None:
        # Convert the image to an OpenCV array (BGR).
        image = np.array(Image.open(image_file))
        if image.ndim == 2:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        if mode in ["Face Detection", "Face Recognition"]:
            net = load_detection_model()
            bboxes = detect_faces_dnn(image.copy(), conf_threshold)
            processing_time = time.time() - start_time
            
            # Draw detection boxes and display confidence.
            for (x1, y1, x2, y2, confidence) in bboxes:
                label = f"{confidence:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # If in Face Recognition mode and known faces are loaded, perform comparison.
            if mode == "Face Recognition" and 'known_files' in locals() and known_files:
                names = recognize_faces(known_encodings, known_names, image.copy(), bboxes)
                for i, (x1, y1, x2, y2, confidence) in enumerate(bboxes):
                    cv2.putText(image, names[i], (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)
            st.markdown(get_image_download_link(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), "result.jpg", "Download Processed Image"), unsafe_allow_html=True)
            st.write(f"Processing time: {processing_time:.2f} seconds")
            
            # Display detection details in an interactive table.
            if bboxes:
                df = pd.DataFrame(bboxes, columns=["x1", "y1", "x2", "y2", "Confidence"])
                st.write("Detection Details:", df)
            else:
                st.info("No faces detected.")
        
        elif mode == "Smile Detection":
            smiles = detect_smiles(image.copy())
            processing_time = time.time() - start_time
            for (x, y, w, h) in smiles:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Smile Detection Result", use_column_width=True)
            st.write(f"Processing time: {processing_time:.2f} seconds")
            st.markdown(get_image_download_link(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), "smile_result.jpg", "Download Processed Image"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()

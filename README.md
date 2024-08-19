# Dental Caries Detection System

## Overview
This project focuses on the development of a Dental Caries Detection System using deep learning models, specifically YOLOv8 and Detectron2. The system is designed to detect caries in bitewing radiographs by identifying and classifying areas of tooth decay. The project aims to enhance dental diagnostics by providing an automated tool to assist in the early detection of dental caries.

## Features
- **Image Upload**: Users can upload a bitewing radiograph from their local device.
- **Image Classification**: The system classifies whether the uploaded image is a dental radiograph.
- **Caries Detection**: If a valid radiograph is detected, the system allows users to run the detection model, which identifies and localizes caries in the image.
- **Result Display**: The system displays the original image with bounding boxes indicating detected caries along with class labels and probabilities.
- **Error Handling**: Non-dental radiographs trigger an error message, and healthy teeth images are recognized and displayed without caries bounding boxes.

## Technology Stack
- **Models**: YOLOv8, Detectron2
- **Frontend**: Streamlit (for web application interface)
- **Backend**: Python (for model training and deployment)
- **Data**: Bitewing radiographs annotated for caries detection

## System Flow
1. **Image Upload**: Users upload a bitewing radiograph.
2. **Image Classification**: The system verifies the uploaded image as a dental radiograph.
3. **Caries Detection**: The system detects caries using the YOLOv8 model and displays the results.
4. **Error Handling**: Non-dental images are rejected with an error message.

## System Evaluation
The system has undergone User Acceptance Testing (UAT) to ensure its usability, accuracy, and reliability. Users reported a user-friendly experience with fast prediction times and effective error handling.

## License
This project is licensed under the MIT License.


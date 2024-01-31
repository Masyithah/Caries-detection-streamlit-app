# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2


# Setting page layout
st.set_page_config(
    page_title="Caries Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Caries Detection using YOLOv8")

model_path_detection = Path(settings.DETECTION_MODEL)
model_path_classification = Path(settings.CLASSIFICATION_MODEL)  

# Load Pre-trained Detection Model
try:
    model_detection = helper.load_model(model_path_detection)
except Exception as ex:
    st.error(f"Unable to load detection model. Check the specified path: {model_path_detection}")
    st.error(ex)

# Load Pre-trained Classification Model
try:
    model_classification = tf.keras.models.load_model(model_path_classification)
except Exception as ex:
    st.error(f"Unable to load classification model. Check the specified path: {model_path_classification}")
    st.error(ex)

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))


# Check if image is uploaded
if uploaded_image is not None:
    try:
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 
        
        # Make predictions using the classification model
        classification_result = model_classification.predict(img_array)

        # Interpret the predictions
        is_xray = classification_result[0, 0] < 0.5
        
        if not is_xray:
            print(classification_result[0, 0])
            st.error("Error: The uploaded image is not a dental radiograph. Please upload the right image.")
        elif st.button("Detect caries"):
            try:
                # Perform object detection
                res = model_detection.predict(PIL.Image.open(uploaded_image), conf=0.2)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                
                # Resize the original image to match the dimensions of the detected image
                resized_original_image = cv2.resize(img_array[0], (res_plotted.shape[1], res_plotted.shape[0]))

                # Display the original image on the left and the detected image on the right
                col1, col2 = st.columns(2)
                col1.image(resized_original_image, caption='Original Image', use_column_width=True, channels="RGB")
                col2.image(res_plotted, caption='Detected Image', use_column_width=True, channels="BGR")

                # Display detection results
                with st.expander("Detection Results"):
                    if boxes:
                        st.write("Decayed teeth")
                    else:
                        st.write("Healthy teeth")
                    # for box in boxes:
                        # st.write(box.data)
            except Exception as ex:
                st.error("Error occurred while processing the image.")
                st.error(ex)
                   
        
    except Exception as ex_classification:
        st.error("Error occurred during image classification.")
        st.error(ex_classification)

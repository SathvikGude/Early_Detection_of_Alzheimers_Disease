import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import cv2
import os

# Load pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for Alzheimer’s disease classification (4 classes)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool the output of the last convolutional layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(4, activation='softmax')(x)  # 4 classes for Alzheimer’s

# Define the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers (we're not training MobileNetV2, only the custom top layers)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (optional, based on your use case)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels for Alzheimer’s disease
CLASS_LABELS = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# Function to preprocess uploaded MRI image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')  # Ensure 3 channels
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Grad-CAM visualization
def grad_cam(image_array, model, layer_name='conv_1'):
    # Ensure image is in batch format
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Print model layer names to check the correct layer name
    for layer in model.layers:
        print(layer.name)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # Use tf.function for optimization (optional but recommended)
    @tf.function
    def compute_gradients(image_array, model, layer_name):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]
        grads = tape.gradient(loss, conv_outputs)[0]
        return conv_outputs, grads

    # Compute gradients and conv outputs
    conv_outputs, grads = compute_gradients(image_array, model, layer_name)

    # Average gradients and apply them to the conv outputs
    weights = tf.reduce_mean(grads, axis=(0, 1))  # Take the mean of the gradients
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / np.max(cam)  # Normalize to [0, 1]

    # Convert to NumPy array and resize
    cam = np.array(cam)
    cam = cv2.resize(cam, (224, 224))  # Resize the heatmap

    return cam

# Overlay Grad-CAM on the original image
def overlay_grad_cam(original_image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# Streamlit App Title
st.title("Early Detection of Alzheimer’s Disease")
st.markdown("This application uses a deep learning model to classify MRI images into four categories:")
st.markdown("- Non-Demented\n- Very Mild Demented\n- Mild Demented\n- Moderate Demented")

# Sidebar for file upload
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(uploaded_file)

    # Predict using the model
    st.write("Analyzing the image...")
    predictions = model.predict(processed_image)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Grad-CAM visualization
    if st.button("Show Grad-CAM Visualization"):
        original_image = np.array(Image.open(uploaded_file).resize((224, 224)))
        heatmap = grad_cam(processed_image, model)
        grad_cam_overlay = overlay_grad_cam(original_image, heatmap)

        st.image(grad_cam_overlay, caption="Grad-CAM Visualization", use_column_width=True)

# Deployment instructions
st.sidebar.markdown("---")
st.sidebar.subheader("How to Use")
st.sidebar.markdown(
    "1. Upload an MRI image in JPG, PNG, or JPEG format.\n"
    "2. Wait for the model to analyze and display the prediction.\n"
    "3. Click 'Show Grad-CAM Visualization' to see regions of interest."
)

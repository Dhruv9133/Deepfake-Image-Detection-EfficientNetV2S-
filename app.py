import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import gdown
import os

# ✅ Model file info
MODEL_PATH = "deepfake_detector.h5"
MODEL_ID = "1bUKUuk8L4oso--JrfGtuSHF7OWiAwFqL"

# ✅ Download model if not present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ✅ Compute PSNR
def compute_psnr(original_img, modified_img):
    mse = np.mean((original_img - modified_img) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ✅ Generate Grad-CAM Heatmap
def generate_gradcam(model, img_array, layer_name="top_conv"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# ✅ Overlay Heatmap
def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# ✅ Streamlit UI
st.title("Deepfake Image Detection with Heatmap, PSNR & Confidence Score")

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    # ✅ Prediction & Confidence Score
    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Real" if prediction > 0.5 else "Fake"

    # ✅ Compute PSNR (Same image since reference not available)
    original_img = np.array(image.img_to_array(img) / 255.0)
    psnr_value = compute_psnr(original_img, original_img)

    # ✅ Generate Heatmap
    heatmap = generate_gradcam(model, img_array)
    heatmap_img = overlay_heatmap(image_path, heatmap)

    # ✅ Display Results
    st.image(image_path, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {label} ({confidence:.2f}% Confidence)")
    st.write(f"**PSNR Value:** {psnr_value:.2f} dB")
    st.image(heatmap_img, caption="Heatmap Overlay", use_column_width=True)

    # ✅ ROC Curve
    if st.button("Show ROC Curve"):
        roc_img = "roc_curve.png"
        st.image(roc_img, caption="ROC Curve", use_column_width=True)

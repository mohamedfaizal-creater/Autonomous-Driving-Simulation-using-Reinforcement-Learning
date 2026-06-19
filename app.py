import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
model = load_model("model/plant_disease_model.h5")

# Load class names
CLASS_NAMES = sorted(os.listdir("dataset/train"))

# Extract leaf type
def get_leaf_name(class_name):
    return class_name.split("_")[0].capitalize()

# Extract disease name
def get_disease_name(class_name):
    parts = class_name.split("_")[1:]
    return " ".join(parts).capitalize()

# UPDATE THESE VALUES from train_model.py output
LEAF_ACCURACY = 95.21   
DISEASE_ACCURACY = 92.77  

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Detection", page_icon="🍃", layout="centered")
st.title("🌿 Plant Leaf Disease Detection System")
st.write("Upload a leaf image to detect the leaf type and disease with confidence level.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
    st.write("🔍 Processing...")

    img = img.resize((128, 128))
    x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    prediction = model.predict(x)[0]
    index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[index]
    leaf_name = get_leaf_name(predicted_class)
    disease_name = get_disease_name(predicted_class)
    confidence = prediction[index] * 100

    st.subheader(f"🍃 Leaf Type: **{leaf_name}**")
    st.subheader(f"🧠 Disease Detected: **{disease_name}**")
    st.subheader(f"📊 Confidence: **{confidence:.2f}%**")

    st.info(f"🌿 Leaf Type Detection Accuracy (Proposed): **{LEAF_ACCURACY}%**")
    st.info(f"🧬 Disease Detection Accuracy (Proposed): **{DISEASE_ACCURACY}%**")

    st.write("### 🔢 Class-wise Confidence:")
    for i, cname in enumerate(CLASS_NAMES):
        st.write(f"- {cname} → {prediction[i] * 100:.2f}%")

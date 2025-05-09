# app.py
import streamlit as st
import pickle
import numpy as np
from PIL import Image

st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

st.title("🐾 Animal Classifier")

# โหลด model และ class names
with open("my_checkpoint.pkl", "rb") as f:
    model, CLASS_NAMES = pickle.load(f)

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ทำ preprocessing ตามที่เทรนไว้
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # พยากรณ์ผล
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.success(f"Predicted: {predicted_class}")

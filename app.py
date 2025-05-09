# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:17:30 2025
@author: hahah
"""

import streamlit as st
from PIL import Image
import numpy as np
import pickle
from pathlib import Path

# --------------------------------------------------------------------
# 1) Model path
MODEL_PATH = 'my_checkpoint.pkl'
CLASS_NAMES = ['cat', 'dog', 'rabbit']  # หรือโหลดจากไฟล์แยกก็ได้

# --------------------------------------------------------------------
# 2) Load model (pickle)
@st.cache_resource
def get_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

model = get_model()

# --------------------------------------------------------------------
# 3) Image preprocessing (เปลี่ยนให้เหมาะกับโมเดลคุณ)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize((224, 224))
    arr = np.array(image) / 255.0
    return arr.reshape(1, -1)  # ปรับตาม input shape ที่โมเดลรับ

# --------------------------------------------------------------------
# 4) Streamlit UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to let the model identify the species.")

uploaded = st.file_uploader("Choose a .jpg / .jpeg / .png image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict_proba(x)[0]  # ต้องใช้ predict_proba สำหรับโมเดลที่รองรับ
        top_k = preds.argsort()[-5:][::-1]

        st.subheader("Prediction (Top‑5)")
        for i in top_k:
            st.write(f"- **{CLASS_NAMES[i]}** : {preds[i]*100:.2f}%")

        with st.expander(f"Show probabilities for all {len(CLASS_NAMES)} species"):
            for i, p in enumerate(preds):
                st.write(f"{CLASS_NAMES[i]:>20} → {p*100:.2f}%")

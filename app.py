import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import pickle
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier")

# โหลดน้ำหนักจากไฟล์ .pkl
MODEL_PATH = 'my_checkpoint.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    model_weights = pickle.load(f)

st.info(f"จำนวนน้ำหนักที่โหลดมา: {len(model_weights)}")

# สร้างโมเดลที่ตรงกับน้ำหนัก (MobileNetV2 แบบเต็ม)
def create_correct_model():
    base_model = MobileNetV2(
        weights=None, 
        include_top=True,  # ใช้ส่วน Classification ด้วย
        input_shape=(224, 224, 3),
        classes=90  # จำนวนคลาสของคุณ
    )
    return base_model

# พยายามสร้างโมเดลและโหลดน้ำหนัก
try:
    model = create_correct_model()
    model.set_weights(model_weights)
    st.success("โหลดโมเดลสำเร็จ!")
except Exception as e:
    st.error(f"ยังเกิดข้อผิดพลาด: {str(e)}")
    st.stop()

# ชื่อคลาส (90 คลาส) - เหมือนเดิม
CLASS_NAMES = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
    # ... (รายชื่อทั้งหมดเหมือนเดิม)
    'zebra'
]

# ส่วนอัพโหลดและทำนาย (เหมือนเดิม)
upload_file = st.file_uploader("อัพโหลดภาพสัตว์:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    img = Image.open(upload_file).resize((224, 224))
    st.image(img, caption="ภาพที่อัพโหลด")
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    predictions = model.predict(x)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    st.success(f"ผลการทำนาย: {predicted_class}")

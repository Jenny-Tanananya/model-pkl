import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import pickle
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier")

# ตรวจสอบไฟล์โมเดล
MODEL_PATH = 'my_checkpoint.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
    st.stop()

# โหลดโมเดลจากไฟล์ .pkl
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # ตรวจสอบประเภทของข้อมูลที่โหลดมา
    if isinstance(loaded_data, list):
        st.warning("ไฟล์นี้เก็บข้อมูลเป็นลิสต์ กำลังพยายามประมวลผล...")
        # ถ้าเป็นลิสต์อาจจะเป็นน้ำหนักของโมเดล (weights)
        # ต้องมีโค้ดสร้างโครงสร้างโมเดลก่อน แล้วค่อยเซ็ตน้ำหนัก
        st.error("ระบบต้องการโครงสร้างโมเดลเพิ่มเติมเพื่อใช้งานลิสต์นี้")
        st.stop()
    elif hasattr(loaded_data, 'predict'):
        model = loaded_data
        st.success("โหลดโมเดลสำเร็จ!")
    else:
        st.error("ไฟล์นี้ไม่ใช่โมเดล Keras ที่สามารถใช้งานได้")
        st.stop()

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {str(e)}")
    st.stop()

# ชื่อคลาส (เหมือนเดิม)
CLASS_NAMES = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 
    'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 
    'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 
    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 
    'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 
    'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
    'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 
    'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

# อัพโหลดไฟล์
upload_file = st.file_uploader("อัพโหลดภาพสัตว์:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    try:
        # แสดงภาพ
        img = Image.open(upload_file)
        st.image(img, caption="ภาพที่อัพโหลด", use_column_width=True)
        
        # ปรับขนาดและเตรียมภาพ
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # ทำนายผล
        predictions = model.predict(x)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = predictions[0][predicted_idx]
        
        # แสดงผลลัพธ์
        st.success(f"ผลการทำนาย: {predicted_class}")
        st.info(f"ความมั่นใจ: {confidence:.2%}")
        
        # แสดง 3 อันดับแรก
        st.subheader("อันดับที่มีความมั่นใจสูงสุด:")
        top_k = 3
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        for i, idx in enumerate(top_indices):
            st.write(f"{i+1}. {CLASS_NAMES[idx]} ({predictions[0][idx]:.2%})")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {str(e)}")

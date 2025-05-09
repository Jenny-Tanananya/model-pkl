import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
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

# โหลดข้อมูลจากไฟล์ .pkl
try:
    with open(MODEL_PATH, 'rb') as f:
        model_weights = pickle.load(f)
    
    if not isinstance(model_weights, list):
        st.error("ไฟล์นี้ไม่เก็บน้ำหนักโมเดลในรูปแบบลิสต์")
        st.stop()

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {str(e)}")
    st.stop()

# สร้างโครงสร้างโมเดล (ต้องตรงกับตอนฝึก)
def create_model():
    # โครงสร้างพื้นฐาน MobileNetV2
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    # เพิ่ม layer สำหรับ classification
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(90, activation='softmax')(x)  # 90 classes
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# สร้างโมเดลและโหลดน้ำหนัก
try:
    model = create_model()
    model.set_weights(model_weights)
    st.success("โหลดโมเดลสำเร็จ!")
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะสร้างโมเดล: {str(e)}")
    st.error("อาจเกิดจาก:")
    st.error("- โครงสร้างโมเดลไม่ตรงกับน้ำหนักที่โหลดมา")
    st.error("- เวอร์ชัน TensorFlow ไม่ตรงกัน")
    st.stop()

# ชื่อคลาส (90 คลาส)
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

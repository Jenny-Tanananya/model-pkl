import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import pickle
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier (EfficientNetB3)")

# โหลดน้ำหนักจากไฟล์ .pkl
MODEL_PATH = 'my_checkpoint.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    model_weights = pickle.load(f)

st.info(f"จำนวนน้ำหนักที่โหลดมา: {len(model_weights)}")

# สร้างโครงสร้าง EfficientNetB3 ที่ตรงกับน้ำหนัก 505 ชุด
def create_effnet_model():
    # โหลด EfficientNetB3 พื้นฐาน (ไม่รวมส่วนหัว)
    base_model = EfficientNetB3(
        weights=None,
        include_top=False,
        input_shape=(300, 300, 3)  # EfficientNetB3 ใช้ขนาดภาพ 300x300
    )
    
    # เพิ่มเลเยอร์แบบกำหนดเอง
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # เลเยอร์เพิ่มเติม
    x = Dense(512, activation='relu')(x)   # เลเยอร์เพิ่มเติม
    predictions = Dense(90, activation='softmax')(x)  # 90 คลาส
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# พยายามสร้างโมเดลและโหลดน้ำหนัก
try:
    model = create_effnet_model()
    
    # ตรวจสอบจำนวนน้ำหนักก่อนโหลด
    st.info(f"จำนวนน้ำหนักที่โมเดลต้องการ: {len(model.get_weights())}")
    
    if len(model.get_weights()) == len(model_weights):
        model.set_weights(model_weights)
        st.success("✅ โหลดโมเดลสำเร็จ!")
    else:
        st.error(f"ยังไม่ตรงกัน! โมเดลต้องการ {len(model.get_weights())} น้ำหนัก")
        st.error("โปรดตรวจสอบว่าโมเดลมีโครงสร้างดังนี้หรือไม่:")
        st.code("""
        1. EfficientNetB3 (include_top=False)
        2. GlobalAveragePooling2D
        3. Dense(1024, relu)
        4. Dense(512, relu)
        5. Dense(90, softmax)
        """)
        st.stop()
        
except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
    st.stop()

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

# ส่วนอัพโหลดและทำนาย
upload_file = st.file_uploader("อัพโหลดภาพสัตว์:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    try:
        # EfficientNetB3 ใช้ขนาดภาพ 300x300
        img = Image.open(upload_file).resize((300, 300))
        st.image(img, caption="ภาพที่อัพโหลด (300x300)")
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        predictions = model.predict(x)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = np.max(predictions[0])
        
        st.success(f"ผลการทำนาย: {predicted_class}")
        st.info(f"ความมั่นใจ: {confidence:.2%}")
        
        # แสดง 3 อันดับแรก
        st.subheader("อันดับที่มีความมั่นใจสูงสุด:")
        top_k = 3
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        for i, idx in enumerate(top_indices):
            st.write(f"{i+1}. {CLASS_NAMES[idx]} ({predictions[0][idx]:.2%})")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะทำนาย: {str(e)}")

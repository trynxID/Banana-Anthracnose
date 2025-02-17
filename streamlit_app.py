import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image

# Load models
@st.cache_resource()
def load_model(model_name):
    return tf.keras.models.load_model(model_name)

# Mapping model names to file paths
model_paths = {
    "CNN": "CNN.h5",
    "VGG16 FFE": "VGG16_FFE.h5",  # Pastikan nama file sesuai
    "VGG16 FT": "VGG16_FT.h5"       # Pastikan nama file sesuai
}

# Streamlit UI
st.title("Deteksi Antraknosa pada Pisang")

# Dropdown untuk memilih model
selected_model_name = st.selectbox("Pilih Model", list(model_paths.keys()))
model = load_model(model_paths[selected_model_name])

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)
    
    # Preprocessing
    image = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Prediksi
    start_time = time.time()
    prediction = model.predict(image_array)[0]
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Menentukan kelas
    class_labels = ["Sehat", "Terinfeksi"]  # Sesuaikan dengan dataset
    predicted_class = class_labels[int(prediction > 0.5)]
    probability = float(prediction) if predicted_class == "Terinfeksi" else 1 - float(prediction)
    
    # Tampilkan hasil
    st.write(f"**Hasil Prediksi:** {predicted_class}")
    st.write(f"**Probabilitas:** {probability:.2%}")
    st.write(f"**Waktu Eksekusi:** {execution_time:.4f} detik")

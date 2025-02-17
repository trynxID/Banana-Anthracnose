import streamlit as st
import tensorflow as tf
import numpy as np
import time
import os
from PIL import Image

# Pastikan daftar model sesuai dengan yang ada di folder utama
model_paths = {
    "CNN": "CNN.h5",
    "VGG16 FFE": "VGG16 FFE.h5",
    "VGG16 FT": "VGG16 FT.h5"
}

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_name):
    if not os.path.exists(model_name):
        st.error(f"Model {model_name} tidak ditemukan! Pastikan file ada di folder utama.")
        st.stop()
    return tf.keras.models.load_model(model_name)

# Streamlit UI
st.title("Deteksi Antraknosa pada Pisang 🍌")
st.write("Upload gambar pisang untuk mendeteksi apakah terkena antraknosa atau tidak.")

# Dropdown untuk memilih model
selected_model_name = st.selectbox("Pilih Model", list(model_paths.keys()))
#model = load_model(model_paths[selected_model_name])
import os

# Cek apakah model ada
st.write("File yang ada di direktori utama:", os.listdir())

# Coba baca model sebagai biner untuk cek korupsi
model_path = model_paths[selected_model_name]
try:
    with open(model_path, "rb") as f:
        f.read(4)  # Coba baca file
    st.write(f"✅ File {model_path} berhasil dibuka.")
except Exception as e:
    st.error(f"❌ Gagal membuka {model_path}: {e}")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png","webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar
    img = image.resize((224, 224))  # Sesuaikan ukuran input model
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension

    # Prediksi
    st.write("⏳ Mendeteksi...")
    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()
    
    # Hasil prediksi
    probability = float(predictions[0][0])
    label = "Terinfeksi Antraknosa" if probability > 0.5 else "Sehat"

    # Waktu eksekusi
    execution_time = round(end_time - start_time, 4)

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"**Kelas:** {label}")
    st.write(f"**Probabilitas:** {probability:.2%}")
    st.write(f"⏱ **Waktu Eksekusi:** {execution_time} detik")

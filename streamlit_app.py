import streamlit as st
import tensorflow as tf
import numpy as np
import time
import os
from PIL import Image

# Cek apakah model tersedia
st.title("Prediksi Antraknosa pada Pisang 🍌")

# Daftar model
model_paths = {
    "CNN": "CNN.h5",
    "VGG16 FFE": "VGG16 FFE.h5",
    "VGG16 FT": "VGG16 FT.h5"
}

# Pilihan model dengan dropdown
selected_model_name = st.selectbox("Pilih Model:", list(model_paths.keys()))

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_name):
    try:
        st.write(f"📂 Memuat model {model_name}...")
        return tf.keras.models.load_model(model_paths[model_name])
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

# Load model berdasarkan pilihan pengguna
model = load_model(selected_model_name)

# Upload file gambar
uploaded_file = st.file_uploader("Unggah gambar pisang 🍌", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar (sesuai dengan input model)
    img = image.resize((224, 224))  # Sesuaikan ukuran input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi dan hitung waktu eksekusi
    st.write("⏳ Memprediksi...")
    start_time = time.time()
    prediction = model.predict(img_array)
    exec_time = time.time() - start_time

    # Menampilkan hasil prediksi
    prob = prediction[0][0]  # Asumsikan output sigmoid (0-1)
    hasil = "Antraknosa" if prob > 0.5 else "Sehat"
    prob_percentage = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)

    st.success(f"✅ Hasil Prediksi: **{hasil}** ({prob_percentage}%)")
    st.write(f"⏱ Waktu eksekusi: {exec_time:.3f} detik")

elif uploaded_file is None:
    st.info("📤 Silakan unggah gambar terlebih dahulu.")

elif model is None:
    st.error("❌ Model tidak ditemukan atau gagal dimuat. Periksa file model Anda!")

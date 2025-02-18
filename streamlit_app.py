import os
import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image

# Judul aplikasi
st.title("Prediksi Antraknosa pada Pisang 🍌")

# Daftar model yang tersedia
model_paths = {
    "CNN": "CNN.keras",
    "VGG16 FFE": "VGG16 FFE.keras",
    "VGG16 FT": "VGG16 FT.keras"
}

# Inisialisasi session state jika belum ada
if "models" not in st.session_state:
    st.session_state["models"] = {}

# Fungsi untuk memuat semua model sekaligus dan menyimpannya dalam dictionary
def load_models():
    for name, path in model_paths.items():
        if not os.path.exists(path):
            st.error(f"⚠ Model {path} tidak ditemukan! Pastikan file ada di folder utama.")
            continue
        try:
            st.write(f"📂 Memuat model {name}...")
            st.session_state["models"][name] = tf.keras.models.load_model(path)
            st.success(f"✅ Model {name} berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ Gagal memuat model {name}: {e}")

# Panggil fungsi pemuatan model (hanya pertama kali)
if not st.session_state["models"]:
    load_models()

# Dropdown pemilihan model
selected_model_name = st.selectbox("Pilih Model:", list(model_paths.keys()))

# Dapatkan model yang sudah dimuat
model = st.session_state["models"].get(selected_model_name)

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar pisang 🍌", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # Baca gambar
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
    prob = predictions[0][0]  # Asumsikan output sigmoid (0-1)
    hasil = "Terinfeksi Antraknosa" if prob > 0.5 else "Sehat"
    prob_percentage = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)
    exec_time = end_time - start_time

    # Menampilkan hasil prediksi
    st.success(f"✅ Hasil Prediksi: **{hasil}** ({prob_percentage}%)")
    st.write(f"⏱ Waktu eksekusi: {exec_time:.3f} detik")
else:
    if uploaded_file is None:
        st.info("📤 Silakan unggah gambar terlebih dahulu.")
    elif model is None:
        st.warning("⚠ Model belum tersedia atau gagal dimuat.")


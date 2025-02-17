import os
import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import threading

# Pastikan daftar model sesuai dengan yang ada di folder utama
st.title("Prediksi Antraknosa pada Pisang 🍌")

# Simpan model yang dipilih sebelumnya
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "model" not in st.session_state:
    st.session_state.model = None

# Daftar model
model_paths = {
    "CNN": "CNN.h5",
    "VGG16 FFE": "VGG16_FFE.h5",  # Pastikan nama file sesuai
    "VGG16 FT": "VGG16_FT.h5"       # Pastikan nama file sesuai
}

# Dropdown pemilihan model
selected_model_name = st.selectbox("Pilih Model:", list(model_paths.keys()))

# Restart aplikasi jika model berubah
if selected_model_name != st.session_state.selected_model:
    st.session_state.selected_model = selected_model_name
    st.session_state.model = None  # Reset model saat model berubah
    st.experimental_rerun()

# Fungsi untuk memuat model
def load_model(model_name):
    if not os.path.exists(model_name):
        st.error(f"Model {model_name} tidak ditemukan! Pastikan file ada di folder utama.")
        return None
    try:
        st.write(f"📂 Memuat model {model_name}...")
        model = tf.keras.models.load_model(model_name)
        st.session_state.model = model
        st.success(f"✅ Model {model_name} berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")

# Cek apakah model ada
st.write("File yang ada di direktori utama:", os.listdir())
model_path = model_paths[selected_model_name]

# Coba baca model sebagai biner untuk cek korupsi
try:
    with open(model_path, "rb") as f:
        f.read(4)  # Coba baca file
    st.write(f"✅ File {model_path} berhasil dibuka.")
except Exception as e:
    st.error(f"❌ Gagal membuka {model_path}: {e}")

# Load model berdasarkan pilihan pengguna secara asinkronus
if st.session_state.model is None:
    load_thread = threading.Thread(target=load_model, args=(model_path,))
    load_thread.start()

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar pisang 🍌", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and st.session_state.model is not None:
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
    predictions = st.session_state.model.predict(img_array)
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
    elif st.session_state.model is None:
        st.warning("🔄 Memuat model, harap tunggu...")

import os
import gdown
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import time

# Membuat folder 'models' jika belum ada
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# URL file Google Drive untuk model
urls = {
    
    "CNN": "https://drive.google.com/uc?id=1-0P6yP2DMDJOaW9aoScbB5qtRFoxgCrD",  # Ganti dengan ID file Google Drive CNN
    "VGG16 FFE": "https://drive.google.com/uc?id=1-0U_idFmsQ5wwm1n3aq46XeCs10j9bmP",  # Ganti dengan ID file Google Drive VGG16 FFE
    "VGG16 FT": "https://drive.google.com/uc?id=1--gWPyFz1fzHJ2LxO2pnjB4C_nzpnLR3",  # Ganti dengan ID file Google Drive VGG16 FT
}

# Mengunduh model ke dalam folder models
for model_name, url in urls.items():
    model_file_path = os.path.join(models_dir, f'{model_name}.h5')
    if not os.path.exists(model_file_path):
        st.write(f"📥 Mengunduh model {model_name}...")
        gdown.download(url, model_file_path, quiet=False)

# Memuat model
model_paths = {
    "CNN": os.path.join(models_dir, "CNN.h5"),
    "VGG16 FFE": os.path.join(models_dir, "VGG16 FFE.h5"),
    "VGG16 FT": os.path.join(models_dir, "VGG16 FT.h5")
}

# Inisialisasi session state jika belum ada
if "models" not in st.session_state:
    st.session_state["models"] = {}

# Fungsi untuk memuat semua model sekaligus dan menyimpannya dalam dictionary
def load_models():
    manage_app_message = st.empty()  # Buat ruang kosong untuk menampilkan pesan
    for name, path in model_paths.items():
        try:
            manage_app_message.text(f"📂 Memuat model {name}...")
            st.session_state["models"][name] = tf.keras.models.load_model(path, compile=False)
            manage_app_message.success(f"✅ Model {name} berhasil dimuat.")
        except Exception as e:
            manage_app_message.error(f"❌ Gagal memuat model {name}: {e}")

# Panggil fungsi pemuatan model (hanya pertama kali)
if not st.session_state["models"]:
    load_models()

# Judul aplikasi Streamlit
st.title("Prediksi Antraknosa pada Pisang 🍌")

# Dropdown pemilihan model
selected_model_name = st.selectbox("Pilih Model:", list(model_paths.keys()))

# Dapatkan model yang sudah dimuat
model = st.session_state["models"].get(selected_model_name)

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar pisang 🍌", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

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
    st.write(f"Model yang digunakan: {selected_model_name}")
    st.success(f"✅ Hasil Prediksi: **{hasil}** ({prob_percentage}%)")
    st.write(f"⏱ Waktu eksekusi: {exec_time:.3f} detik")
else:
    if uploaded_file is None:
        st.info("📤 Silakan unggah gambar terlebih dahulu.")

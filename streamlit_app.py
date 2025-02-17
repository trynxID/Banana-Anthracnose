import os
from PIL import Image

# Pastikan daftar model sesuai dengan yang ada di folder utama
st.title("Prediksi Antraknosa pada Pisang 🍌")
# Simpan model yang dipilih sebelumnya
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
# Daftar model
model_paths = {
    "CNN": "CNN.h5",
    "VGG16 FFE": "VGG16 FFE.h5",
    "VGG16 FT": "VGG16 FT.h5"
}

# Dropdown pemilihan model
selected_model_name = st.selectbox("Pilih Model:", list(model_paths.keys()))
# Restart aplikasi jika model berubah
if selected_model_name != st.session_state.selected_model:
    st.session_state.selected_model = selected_model_name
    st.rerun()
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
    try:
        st.write(f"📂 Memuat model {model_name}...")
        return tf.keras.models.load_model(model_paths[model_name])
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

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
# Load model berdasarkan pilihan pengguna
model = load_model(selected_model_name)

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png","webp"])
# Upload file gambar
uploaded_file = st.file_uploader("Unggah gambar pisang 🍌", type=["jpg", "png", "jpeg","webp"])

if uploaded_file:
if uploaded_file is not None and model is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.image(image, caption="📷 Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar
    # Preprocessing gambar (sesuai dengan input model)
    img = image.resize((224, 224))  # Sesuaikan ukuran input model
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    st.write("⏳ Mendeteksi...")
    # Prediksi dan hitung waktu eksekusi
    st.write("⏳ Memprediksi...")
    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()
    
    # Hasil prediksi
    probability = float(predictions[0][0])
    label = "Terinfeksi Antraknosa" if probability > 0.5 else "Sehat"
    prediction = model.predict(img_array)
    exec_time = time.time() - start_time
    # Menampilkan hasil prediksi
    prob = prediction[0][0]  # Asumsikan output sigmoid (0-1)
    hasil = "Antraknosa" if prob > 0.5 else "Sehat"
    prob_percentage = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)
    st.success(f"✅ Hasil Prediksi: **{hasil}** ({prob_percentage}%)")
    st.write(f"⏱ Waktu eksekusi: {exec_time:.3f} detik")

    # Waktu eksekusi
    execution_time = round(end_time - start_time, 4)
elif uploaded_file is None:
    st.info("📤 Silakan unggah gambar terlebih dahulu.")

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"**Kelas:** {label}")
    st.write(f"**Probabilitas:** {probability:.2%}")
    st.write(f"⏱ **Waktu Eksekusi:** {execution_time} detik")
elif model is None:
    st.error("❌ Model tidak ditemukan atau gagal dimuat. Periksa file model Anda!")

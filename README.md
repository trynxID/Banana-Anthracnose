# Deteksi Antraknosa pada Pisang Menggunakan Transfer Learning dengan VGG16

## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model deteksi penyakit antraknosa pada pisang menggunakan pendekatan *transfer learning* dengan arsitektur VGG16. Model dikembangkan dan diuji menggunakan dataset citra pisang yang telah dikurasi, serta dibandingkan dengan model *Convolutional Neural Network* (CNN) konvensional untuk menilai efektivitas pendekatan *transfer learning*.

## Struktur Direktori

### Model Jaringan Syaraf Tiruan (Neural Network)
- **`CNN.h5`**: Model CNN dalam format HDF5.
- **`CNN.keras`**: Model CNN dalam format Keras.
- **`VGG16 FFE.h5`**: Model VGG16 dengan metode *Fixed Feature Extraction (FFE)* dalam format HDF5.
- **`VGG16 FFE.keras`**: Model VGG16 FFE dalam format Keras.
- **`VGG16 FT.h5`**: Model VGG16 yang telah melalui proses *fine-tuning* dalam format HDF5.
- **`VGG16 FT.keras`**: Model VGG16 *fine-tuned* dalam format Keras.

### Dataset
- **`Dataset_Pisang.zip`**: Dataset citra pisang yang digunakan untuk pelatihan dan pengujian model.

### Skrip dan Notebook
- **`pengujian_3_model_dense_32_dropout_50_ada...`**: Skrip atau notebook yang digunakan untuk menguji tiga model dengan arsitektur tertentu (*Dense* 32, *Dropout* 50%).
- **`streamlit_app.py`**: Aplikasi berbasis *Streamlit* untuk implementasi model dalam bentuk antarmuka web interaktif.

### Dependensi dan Konfigurasi
- **`requirements.txt`**: Daftar pustaka Python yang dibutuhkan untuk menjalankan proyek.

## Instalasi
1. **Kloning repositori**
   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```
2. **Buat dan aktifkan lingkungan virtual (opsional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk macOS/Linux
   venv\Scripts\activate     # Untuk Windows
   ```
3. **Instal dependensi**
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi Streamlit (opsional)**
   ```bash
   streamlit run streamlit_app.py
   ```

## Penggunaan
- Gunakan model yang tersedia untuk melakukan klasifikasi citra pisang sehat dan yang terinfeksi antraknosa.
- Gunakan aplikasi *Streamlit* untuk melakukan inferensi berbasis antarmuka pengguna.
- Modifikasi dan latih ulang model jika diperlukan untuk meningkatkan akurasi.

## Kontributor
- Nama Anda : Muhammad Sidiq Firmansyah
- Kontak atau akun GitHub Anda : TrynxID

---
Untuk pertanyaan atau saran, silakan buat *issue* atau hubungi saya melalui GitHub.


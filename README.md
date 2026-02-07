# Analisis Sentimen Program MBG (Makan Bergizi Gratis)

Dashboard analisis sentimen berbahasa Indonesia untuk menganalisis opini publik terhadap Program Makan Bergizi Gratis (MBG) menggunakan Support Vector Machine (SVM).

## Deskripsi

Aplikasi web berbasis Streamlit untuk melakukan analisis sentimen terhadap teks berbahasa Indonesia, khususnya terkait Program MBG. Aplikasi ini menggunakan algoritma Support Vector Machine (SVM) dengan preprocessing teks yang komprehensif.

## Fitur Utama

### 1. Dashboard Utama
- Visualisasi distribusi sentimen (pie chart)
- Word cloud untuk sentimen positif dan negatif
- Top 15 kata teratas per sentimen
- Tampilan dataset training lengkap dengan text wrapping

### 2. Prediksi Sentimen
- Input teks manual untuk prediksi real-time
- Menampilkan confidence score
- Demo preprocessing step-by-step

### 3. Analisis CSV Batch
- Upload file CSV untuk analisis massal
- Deteksi otomatis kolom tanggal
- Visualisasi temporal (distribusi per tahun dan bulan)
- Filter berdasarkan confidence threshold
- Export hasil ke CSV
- Interpretasi dan rekomendasi otomatis

### 4. Demo Preprocessing
- Visualisasi tahapan preprocessing:
  - Text Cleaning
  - Case Folding
  - Tokenization
  - Normalization
  - Stopwords Removal
  - Stemming

## Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. Clone repository:
```bash
git clone https://github.com/username/analisis-sentimen-mbg.git
cd analisis-sentimen-mbg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (jika belum ada):
```python
python -c "import nltk; nltk.download('punkt')"
```

## Struktur File

```
analisis-sentimen-mbg/
├── dashboard.py              # Aplikasi Streamlit utama
├── sentiment_model.py        # Model SVM dan preprocessing
├── sentiment_model.pkl       # Model terlatih
├── kolom_lengkap.csv        # Dataset training
├── requirements.txt          # Dependencies Python
├── generate_test_data.py     # Untuk generate data test csv
├── README.md                # Dokumentasi ini
```

## Cara Penggunaan

### Menjalankan Dashboard

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

### Menggunakan Fitur Prediksi

1. Pilih menu **"Prediksi Sentimen"**
2. Masukkan teks dalam bahasa Indonesia
3. Klik **"Prediksi Sentimen"**
4. Lihat hasil prediksi dan confidence score

### Analisis CSV Batch

1. Pilih menu **"📁 Analisis CSV"**
2. Upload file CSV dengan kolom `text` atau `full_text`
3. Atur confidence threshold (default: 0.5)
4. Klik **"🚀 Proses Prediksi Sentimen"**
5. Lihat hasil analisis dan visualisasi
6. Download hasil jika diperlukan

### Format CSV untuk Upload

File CSV harus memiliki minimal kolom berikut:
- `text` atau `full_text`: Kolom berisi teks untuk dianalisis

Opsional:
- `created_at` atau kolom tanggal lainnya untuk analisis temporal

Contoh:
```csv
text,created_at
"Program MBG sangat membantu anak-anak",2025-07-06 10:30:00
"MBG tidak efektif dan boros anggaran",2025-07-07 14:20:00
```

## Preprocessing Pipeline

1. **Text Cleaning**: Menghapus URL, mention, hashtag, angka, dan karakter khusus
2. **Case Folding**: Mengubah semua teks menjadi lowercase
3. **Tokenization**: Memecah teks menjadi token/kata
4. **Normalization**: Menormalkan kata-kata tidak baku
5. **Stopwords Removal**: Menghapus kata-kata umum yang tidak bermakna
6. **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan Sastrawi

## Model

- **Algoritma**: Support Vector Machine (SVM) dengan kernel linear
- **Features**: TF-IDF Vectorization
- **Dataset**: 1,906 data tweet tentang Program MBG
  - Positif: 793 data
  - Negatif: 1,113 data

## Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.0
- Sastrawi >= 1.0.1
- plotly >= 5.17.0
- wordcloud >= 1.9.0
- matplotlib >= 3.7.0
- python-dateutil >= 2.8.0

Lihat `requirements.txt` untuk daftar lengkap.

## Catatan Penting

### Confidence Threshold
- Data dengan confidence score di bawah threshold tidak dimasukkan dalam analisis utama
- Default threshold: 0.5 (50%)
- Dapat disesuaikan melalui slider di halaman Analisis CSV

### Text Wrapping
- Semua tabel menggunakan text wrapping otomatis
- Teks panjang akan ditampilkan lengkap tanpa terpotong

### Analisis Temporal
- Hanya tahun valid (1900-2100) yang ditampilkan di chart
- Deteksi otomatis berbagai format tanggal

## Lisensi

[MIT License](LICENSE)


## Acknowledgments

- Dataset: Twitter/X data tentang Program MBG
- Sastrawi: Indonesian stemmer library
- NLTK: Natural Language Toolkit
- Streamlit: Framework untuk web app

---

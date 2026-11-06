**“Klasifikasi Gambar Asli dan Gambar Sintetis AI Menggunakan Convolutional Neural Network pada Dataset CIFAKE.”**

Implementasi tugas individu mata kuliah Deep Learning. Model CNN dilatih dari awal untuk membedakan **REAL** (gambar asli) dan **FAKE** (gambar sintetis AI) pada CIFAKE. Masalah ini merupakan klasifikasi seluruh gambar menjadi dua kelas; model tidak menghasilkan kotak lokasi objek.

**Status paket awal:** kode dan notebook tersedia. Pelatihan penuh CIFAKE belum dijalankan dan belum ada angka hasil eksperimen. Jalankan notebook untuk menghasilkan model serta metrik yang akan dilaporkan. Hasil pengujian teknis pada data buatan bukan hasil eksperimen CIFAKE.

## Tujuan dan batas penelitian

Pertanyaan penelitian: seberapa baik CNN sederhana membedakan citra asli dan citra sintetis dalam dataset CIFAKE?

CNN mempelajari fitur spasial melalui konvolusi. Pada tugas ini, fitur dapat mencakup tekstur dan pola lokal yang berbeda antara dua sumber data. Keberadaan fitur tersebut tidak membuktikan bahwa model memahami proses pembuatan gambar.

Eksperimen mengukur kinerja dalam distribusi CIFAKE. REAL berasal dari CIFAR-10 dan FAKE dibuat dengan Stable Diffusion 1.4. Model mungkin mempelajari karakteristik dataset, kompresi, atau generator tertentu. Hasilnya tidak dapat langsung digeneralisasi ke seluruh gambar dari generator lain, gambar beresolusi tinggi, gambar yang diedit, atau gambar dari media sosial. Skor sigmoid belum dikalibrasi dan bukan bukti pasti keaslian suatu gambar.

## Dataset

Sumber utama: [repository penulis CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) dan [halaman dataset Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

| Bagian resmi | REAL | FAKE | Total |
|---|---:|---:|---:|
| Train | 50.000 | 50.000 | 100.000 |
| Test | 10.000 | 10.000 | 20.000 |
| Total | 60.000 | 60.000 | 120.000 |

Citra RGB berukuran 32×32 piksel. Label dibuat eksplisit: **REAL=0, FAKE=1**; skor keluaran model mengarah pada FAKE. Jangan mengandalkan urutan alfabet folder untuk menentukan label.

Validasi diambil secara terstratifikasi sebesar 20% dari bagian train resmi. Sebelum audit, rancangan pembagiannya adalah 80.000 train, 20.000 validation, dan 20.000 test. Jumlah aktual dapat berubah setelah gambar train yang identik disisihkan.

Audit menghitung SHA256 dari piksel RGB yang didekode. Gambar train yang identik dengan test disisihkan dari train; pengulangan piksel identik di train disatukan sebelum pemisahan train/validation. Test resmi dipertahankan seluruhnya, termasuk pengulangan internal yang jumlahnya dicatat. Label yang bertentangan dan gambar rusak menghentikan proses. Pemeriksaan ini belum mendeteksi gambar yang hanya mirip atau turunan hasil crop/kompresi.

Data mentah tidak disertakan dalam paket ini. Notebook mengunduh data dari Kaggle. Rujukan dan ketentuan penggunaan data tetap mengikuti sumber aslinya; penulis CIFAKE menyatakan lisensi MIT dan meminta sitasi CIFAKE serta CIFAR-10.

## Menjalankan di Google Colab

1. Buka [Google Colab](https://colab.research.google.com/) dan unggah `CNN_Deteksi_Gambar_AI.ipynb` melalui menu pembukaan notebook.
2. Pilih runtime dengan GPU bila tersedia. CPU tetap bisa dipakai, tetapi pelatihan lebih lama.
3. Isi identitas pada sel pembuka. Jalankan sel dari atas ke bawah. Notebook berdiri sendiri, sehingga tidak perlu mengunggah file Python secara terpisah.
4. Secara default, data diunduh lewat `kagglehub`. Jika Kaggle meminta autentikasi, gunakan login resmi di sel opsional atau unduh dataset sendiri, ekstrak, dan isi `LOCAL_DATA_DIR`. Jangan menaruh token atau `kaggle.json` dalam repository.
5. Tunggu pelatihan dan evaluasi selesai. Sel terakhir mengunduh ZIP hasil agar tetap tersimpan setelah sesi Colab berakhir. Unduh juga notebook beserta output selnya.
6. Baca metrik dan kesalahan prediksi, lalu tulis pembahasan dengan angka yang benar-benar dihasilkan. Nama folder hasil harus baru/kosong untuk setiap eksperimen.

Notebook menggunakan TensorFlow yang tersedia pada runtime Colab; instalasi tambahan hanya memasang dependensi yang belum memenuhi kebutuhan. Setiap run menyimpan versi paket aktual pada `run_config.json`. Seed membantu reproduksi, tetapi versi library dan perangkat yang berbeda dapat menghasilkan perbedaan numerik.

## Menjalankan secara lokal

Gunakan Python 3.10–3.12 dan lingkungan virtual. Dari folder proyek:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python cnn_detector.py train --output-dir results/cifake_01
```

Untuk dataset yang sudah diunduh:

```bash
python cnn_detector.py train --data-dir data/cifake --epochs 20 --output-dir results/cifake_01
```

Struktur yang dibutuhkan:

```text
data/cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

Prediksi satu gambar setelah pelatihan:

```bash
python cnn_detector.py predict --model results/cifake_01/best_model.keras --image contoh.jpg
```

Input baru diubah menjadi RGB 32×32 dengan interpolasi bilinear. Perubahan ukuran ini membuang banyak detail; contoh dari luar CIFAKE perlu diperlakukan sebagai demonstrasi dan tidak digabungkan dengan evaluasi test resmi.

## Arsitektur dan pelatihan

| Lapisan | Bentuk keluaran |
|---|---|
| Input RGB | 32×32×3 |
| Rescaling 1/255 | 32×32×3 |
| Conv2D 32, kernel 3×3, ReLU, same | 32×32×32 |
| MaxPooling 2×2 | 16×16×32 |
| Conv2D 64, kernel 3×3, ReLU, same | 16×16×64 |
| MaxPooling 2×2 | 8×8×64 |
| Conv2D 128, kernel 3×3, ReLU, same | 8×8×128 |
| MaxPooling 2×2 | 4×4×128 |
| Flatten | 2.048 |
| Dense 128, ReLU | 128 |
| Dropout 0,5 | 128 |
| Dense 1, sigmoid | Skor FAKE |

Total parameter: **355.649**, seluruhnya dapat dilatih. Ini merupakan rancangan baseline proyek, bukan reproduksi persis arsitektur terbaik makalah CIFAKE.

Adam memakai learning rate 0,001, batch size 128, maksimum 20 epoch, dan loss binary cross-entropy. Early stopping memakai validation loss dengan patience 4. Checkpoint dipilih berdasarkan validation loss terendah. Threshold tetap 0,5 ditetapkan sebelum evaluasi test: skor ≥0,5 diklasifikasikan FAKE. Tidak ada augmentasi pada baseline agar pengaruh transformasi tambahan tidak tercampur pada percobaan pertama. Model tidak memakai bobot pralatih.

## Evaluasi dan berkas hasil

FAKE diperlakukan sebagai kelas positif. Precision FAKE mengukur proporsi prediksi FAKE yang benar. Recall FAKE mengukur bagian gambar FAKE yang berhasil ditemukan. F1 menyeimbangkan precision dan recall. ROC-AUC dihitung dari skor kontinu.

| Berkas | Fungsi |
|---|---|
| `best_model.keras` | Model dengan validation loss terbaik, termasuk normalisasi |
| `metrics.json` | Accuracy, precision/recall/F1 FAKE, ROC-AUC, majority baseline |
| `classification_report.txt` | Precision, recall, F1, dan support untuk kedua kelas |
| `confusion_matrix.png` | Baris: label sebenarnya; kolom: prediksi; urutan REAL, FAKE |
| `training_curves.png`, `roc_curve.png` | Grafik pelatihan dan evaluasi |
| `test_predictions.csv` | Prediksi setiap gambar test dalam urutan tetap |
| `error_examples.png` | Maksimum 8 kesalahan pertama; hanya dibuat bila ada kesalahan |
| `training_history.csv` | Riwayat metrik per epoch |
| `data_audit.json`, `*_manifest.csv` | Hitungan audit serta pembagian data yang digunakan |
| `run_config.json`, `model_summary.txt` | Konfigurasi, versi library, dan arsitektur |

Baseline mayoritas ditentukan hanya dari label train lalu dievaluasi pada test. Untuk data seimbang, accuracy sekitar 50% merupakan pembanding dasar, bukan target hasil CNN. Test dipakai setelah pemilihan model selesai. Apabila mencoba arsitektur lain, pilih berdasarkan validation dan hindari menyesuaikan model berulang kali terhadap test.

## Pembahasan setelah eksperimen

Isi bagian ini setelah pelatihan menggunakan berkas hasil:

- Jumlah data aktual tiap split dan jumlah duplikat yang disisihkan.
- Epoch terbaik berdasarkan validation loss, serta pola kurva train/validation.
- Accuracy, precision FAKE, recall FAKE, F1 FAKE, ROC-AUC, serta baseline mayoritas.
- Jumlah REAL yang salah menjadi FAKE dan FAKE yang lolos sebagai REAL dari confusion matrix.
- Contoh kesalahan dan dugaan penyebabnya. Nyatakan dugaan sebagai interpretasi, bukan fakta yang sudah dibuktikan.
- Keterbatasan dataset dan generalisasi, serta satu usulan perbaikan yang dapat diuji.

Jangan menyalin angka akurasi penelitian terdahulu sebagai hasil proyek sendiri.

## Mengumpulkan melalui GitHub

1. Buat repository, misalnya `cnn-ai-image-detector`.
2. Unggah isi folder proyek ke akar repository: notebook, `cnn_detector.py`, `requirements.txt`, `.gitignore`, dan README ini. Isi identitas sendiri.
3. Jalankan pelatihan penuh. Simpan notebook dengan output serta hasil eksperimen pada `results/cifake_01/`. Jika model terlalu besar untuk unggahan lewat browser GitHub, gunakan Git atau sertakan tautan unduh model yang dapat diakses penilai.
4. Lengkapi pembahasan hasil pada README. Pastikan status proyek di bagian atas diperbarui sesuai pelaksanaan nyata.
5. Periksa repository bisa diakses penilai, lalu kumpulkan URL repository di Google Spreadsheet dari dosen/asisten.

Jangan mengunggah seluruh dataset, cache, atau kredensial. Hasil teknis uji pipeline tidak menggantikan eksperimen penuh.

## Referensi

- Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access. [DOI: 10.1109/ACCESS.2024.3356122](https://doi.org/10.1109/ACCESS.2024.3356122); [preprint](https://arxiv.org/abs/2303.14126).
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. [Laporan teknis CIFAR](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf); [halaman CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Dokumentasi klasifikasi gambar TensorFlow](https://www.tensorflow.org/tutorials/images/classification).
- [Dokumentasi KaggleHub](https://github.com/Kaggle/kagglehub).

# Bab 5 — Image Classification

Kode Example 5.1–5.16 dan Exercise 5.1–5.22 dari buku Perry Xiao, *Artificial Intelligence
Programming with Python*, Bab 5. Semua kode ada di `Code/`, hasil (gambar/CSV) ada di `Output/`.

## Cara menjalankan

Buka terminal di folder `Bab 5/Code` dengan venv `.venv311` aktif:

```
python example5.2.py                # script biasa
streamlit run example5.9a.py        # aplikasi web (Example 5.9, 5.9a, 5.10, 5.10a, Exercise 5.12-5.14)
```

Bobot model Keras disimpan di `D:\AI\.keras` dan dataset Kaggle di `D:\AI\.kagglehub`
(bukan di C). Pertama kali sebuah model dipakai, bobotnya diunduh dulu.

## Isi folder `Code/`

| File / folder | Keterangan |
|---|---|
| `example5.*.py`, `exercise5.*.py` | kode tiap example dan exercise |
| `transfer_utils.py` | fungsi bersama untuk Exercise 5.5–5.10 (load dataset, buat model, training, tabel perbandingan) |
| `Flowers/` | dataset 5 kelas bunga × 40 gambar (pengganti dataset `SkinArea` buku yang tidak publik) |
| `models/` | model super resolution (EDSR, ESPCN, FSRCNN, LapSRN) dan model pewarnaan foto |
| `saved_models/` | model hasil training Example 5.6–5.8 |
| `streamlit-file-image-classification/` | repo siap upload ke GitHub untuk Exercise 5.15–5.16 |
| `*.jpg`, `*.jpeg`, `sample_video.mp4` | gambar uji (Wikimedia Commons / OpenCV sample) dan video contoh |

## Perbedaan dengan kode di buku

Kode buku ditulis untuk TensorFlow 2.4. Di TensorFlow 2.20 / Keras 3 beberapa hal harus diubah:

- `fit_generator` / `evaluate_generator` dihapus → diganti `fit` / `evaluate`.
- `st.cache` (Streamlit) dihapus → diganti `st.cache_resource`; `use_column_width` → `width="stretch"`.
- `np.int`, `cv2.cv2`, `generator.next()` sudah tidak ada → `int`, `cv2`, `next(generator)`.
- Beberapa bug di buku ikut diperbaiki: Example 5.2 memprediksi `x` mentah (seharusnya `processed_image`),
  gambar BGR dari OpenCV tidak diubah ke RGB sebelum masuk model, Example 5.6h lupa preprocessing,
  Example 5.8 memakai `sigmoid` dengan `categorical_crossentropy` (diganti `softmax`), Example 5.16
  membuat array `np.int8` sehingga hasil filter terpotong (diganti `float32`).
- Gambar London (Pexels) untuk stitching tidak bisa diunduh → diganti `boat1-3.jpg` dari OpenCV.

## Hasil dan komentar

### Example

| Example | Hasil |
|---|---|
| 5.2 | MobileNet menebak `Elephant.jpg` sebagai **African_elephant (74,8%)**. |
| 5.6 | VGG16 + Flatten + 2×Dense(1024): akurasi uji **58%** (buku: 39%). Akurasi training 100% tapi validasi turun → *overfitting*, karena kepala model punya ±26 juta parameter sementara gambar training hanya 105. |
| 5.7 | ImageDataGenerator + augmentasi: akurasi uji **76,7%**, lebih baik dari 5.6 karena augmentasi dan kepala model lebih kecil (128 neuron). |
| 5.8 | CNN buatan sendiri tanpa bobot pre-trained: akurasi uji **36,7%**. Data terlalu sedikit untuk belajar dari nol, dan ini menunjukkan kenapa transfer learning penting. |
| 5.11 | `boat1 + boat2` berhasil digabung menjadi satu panorama. |
| 5.12 | Area yang dipilih hilang dan terisi warna sekitarnya (lihat `Output/example 5.12.jpg`). |
| 5.14–5.16 | Kernel Gabor, hasil filter nanas, dan efek Fractalius (lihat `Output/`). |

Example 5.3, 5.4, 5.5, 5.10, 5.10a dan 5.13 memakai webcam (kamera #0) dan harus dijalankan
langsung di laptop. Example 5.9, 5.9a, 5.10, 5.10a dijalankan dengan `streamlit run`.

### Exercise 5.1 – 5.2: memilih model dengan if-else

| Model | Layer | Parameter |
|---|---:|---:|
| VGG16 | 23 | 138.357.544 |
| VGG19 | 26 | 143.667.240 |
| ResNet50 | 177 | 25.636.712 |
| InceptionV3 | 313 | 23.851.784 |
| DenseNet201 | 709 | 20.242.984 |
| EfficientNetB7 | 816 | 66.658.687 |
| EfficientNetB0 | 240 | 5.330.571 |
| MobileNetV2 | 156 | 3.538.984 |

Komentar: jumlah layer tidak sama dengan ukuran model. VGG hanya punya ±25 layer tapi parameternya
paling banyak, karena layer Dense di akhirnya sangat besar. DenseNet201 punya 709 layer tapi
parameternya 7× lebih sedikit dari VGG16. MobileNet dan EfficientNetB0 dirancang ringan untuk HP.

### Exercise 5.3: membandingkan model pada gambar hewan

| Gambar | VGG16 | ResNet50 | MobileNet | InceptionV3 |
|---|---|---|---|---|
| Elephant.jpg | African_elephant 0,90 | African_elephant 0,91 | African_elephant 0,75 | African_elephant 0,79 |
| cat.jpg | Egyptian_cat 0,52 | Egyptian_cat 0,42 | tiger_cat 0,52 | tabby 0,81 |
| bird2.jpeg | toucan 0,86 | toucan 0,96 | toucan 1,00 | toucan 0,96 |
| baboon.jpg | baboon 0,96 | baboon 0,71 | baboon 0,98 | baboon 0,94 |

Komentar: keempat model benar untuk gajah, toucan, dan baboon. Untuk kucing, jawabannya berbeda-beda
(Egyptian_cat, tiger_cat, tabby) karena ImageNet membedakan jenis kucing yang mirip satu sama lain.
Tingkat keyakinan untuk kucing juga paling rendah. MobileNet paling cepat (±0,3 detik per gambar)
dengan akurasi yang tetap bagus.

### Exercise 5.4: klasifikasi dari file video

Dengan `sample_video.mp4` (gajah → kucing → toucan → jeruk → apel), MobileNet mengenali semua objek
dengan kecepatan ±9 FPS di CPU. Jalankan `python exercise5.4.py --model vgg16` untuk webcam.

### Exercise 5.5 – 5.6: transfer learning dengan berbagai base model (dataset Flowers, 10 epoch)

| Base model | Waktu training (s) | Akurasi uji | Parameter |
|---|---:|---:|---:|
| VGG16 | 538 | 0,70 | 14,8 jt |
| ResNet50 | 444 | 0,84 | 24,1 jt |
| MobileNetV2 | 152 | 0,86 | 2,6 jt |
| DenseNet201 | 652 | 0,84 | 18,8 jt |
| InceptionV3 | 453 | 0,82 | 22,3 jt |
| VGG19 | 941 | 0,82 | 20,2 jt |
| DenseNet121 | 207 | 0,86 | 7,3 jt |
| Xception | 726 | 0,80 | 21,4 jt |
| EfficientNetB0 | 686 | 0,86 | 4,4 jt |
| **MobileNetV3Large** | **67** | **0,86** | 3,2 jt |
| **ConvNeXtTiny** | 1586 | **0,88** | 28,0 jt |

Komentar: semua model memakai kepala yang sama (GlobalAveragePooling + Dense 256), jadi hasilnya
bisa dibandingkan dengan adil. ConvNeXtTiny (model terbaru) paling akurat tapi juga paling lambat.
MobileNetV3Large memberi akurasi hampir sama (0,86) dalam waktu 67 detik, 23× lebih cepat, sehingga
paling efisien. VGG16 paling lemah karena arsitekturnya paling tua. Model besar tidak selalu lebih
akurat: dengan data sekecil ini (150 gambar training), perbedaan akurasi antar model hanya beberapa persen.

### Exercise 5.7 – 5.10: citra medis (dataset Kaggle, diunduh otomatis lewat `kagglehub`)

| Exercise | Dataset | Model | Akurasi uji | Waktu (s) |
|---|---|---|---:|---:|
| 5.7 | Kanker kulit (benign / malignant) | VGG16 | 0,830 | 1528 |
| | | ResNet50 | 0,805 | 504 |
| | | DenseNet201 | 0,840 | 980 |
| 5.8 | Retinopati diabetik (5 kelas) | VGG16 | 0,480 | 2102 |
| | | MobileNetV2 | 0,527 | 142 |
| | | DenseNet121 | 0,533 | 582 |
| 5.9 | X-ray COVID-19 (covid / normal) | VGG16 | 0,958 | 126 |
| | | MobileNet | 0,958 | 14 |
| 5.10 | MRI tumor otak (no / yes) | VGG16 | 0,875 | 347 |
| | | InceptionV3 | 0,875 | 243 |

Komentar:
- **5.7**: VGG16 bersaing dengan ResNet50 dan DenseNet201 (80–84%), tapi training-nya paling lama.
  Dataset HAM10000 dari buku (2,7 GB) diganti dataset ISIC benign/malignant yang lebih kecil; dipakai
  300 gambar training dan 100 gambar uji per kelas.
- **5.8**: paling sulit karena ada 5 tingkat keparahan yang perbedaannya halus. Akurasi ±50%
  (tebakan acak 20%). VGG16 paling lambat dan paling rendah. Untuk hasil lebih baik perlu lebih banyak
  data (di sini hanya 120 per kelas) dan *fine-tuning* layer atas base model.
- **5.9**: MobileNet sama akuratnya dengan VGG16 (95,8%) tapi 9× lebih cepat, sehingga lebih cocok untuk HP.
- **5.10**: dataset RSNA dari buku berukuran 427 GB sehingga tidak mungkin diunduh. Exercise ini
  dikerjakan dengan dataset MRI tumor otak dari subbab 5.4.5. VGG16 dan InceptionV3 sama-sama 87,5%,
  tapi InceptionV3 lebih cepat.

### Exercise 5.11: federated learning (MNIST, 10 klien, 10 ronde)

Akurasi model global naik dari 16% (ronde 1) ke **88,1%** (ronde 10), sedangkan training terpusat
mencapai **97,2%**. Federated learning sedikit lebih rendah karena tiap klien hanya punya data
2 digit (non-IID), tapi data tidak pernah keluar dari klien. Package `tensorflow-federated` tidak
mendukung Windows, jadi algoritma FedAvg ditulis manual dengan Keras.

### Exercise 5.12 – 5.14: aplikasi Streamlit

- 5.12: aplikasi upload gambar + VGG19, sekarang juga menampilkan tabel top-3.
- 5.13: pemutar video dari file (`sample_video.mp4` atau upload) dengan slider kontras dan kecerahan.
- 5.14: klasifikasi webcam / video + VGG19, menampilkan FPS. VGG19 paling berat sehingga FPS-nya
  paling rendah; MobileNet paling lancar.

### Exercise 5.15 – 5.16: GitHub dan Streamlit Cloud

Folder `streamlit-file-image-classification/` sudah siap diupload (berisi kode, `requirements.txt`,
`README.md`). Langkah membuat repository GitHub dan deploy ke share.streamlit.io ada di README folder
tersebut. Langkah ini harus dilakukan sendiri karena memakai akun GitHub milikmu.

### Exercise 5.17: stitching tiga gambar

`boat1-3.jpg` berhasil digabung: mode PANORAMA → 1626×681 piksel, mode SCANS → 1686×703 piksel.
PANORAMA memproyeksikan gambar ke permukaan silinder/bola (cocok untuk kamera yang diputar),
sedangkan SCANS menganggap gambar datar (cocok untuk hasil scan dokumen).

### Exercise 5.18: inpainting

Menara TV dan satu awan di `church.jpeg` dihapus (`python exercise5.18.py --auto`). Kedua algoritma
berhasil karena latarnya langit yang polos. Telea sedikit lebih halus, Navier-Stokes kadang
meninggalkan garis samar. Inpainting bekerja baik untuk objek tipis di latar seragam, tapi kurang
baik untuk area besar dengan banyak tekstur.

### Exercise 5.19: mewarnai foto hitam-putih

Model Zhang et al. (Colorful Image Colorization) dijalankan lewat OpenCV DNN. Untuk demo, foto berwarna
diubah ke hitam-putih lalu diwarnai lagi. Langit, rumput, dan pepohonan diwarnai dengan benar. Benda
yang warnanya "bebas" (baju, bangunan) cenderung kecokelatan atau pucat, karena model menebak warna
yang paling umum. PySimpleGUI 5 sekarang berbayar, jadi versi GUI dari GitHub tidak dipakai.
Untuk foto lama sendiri: `python exercise5.19.py foto_lama.jpg`.

### Exercise 5.20: super resolution dengan berbagai model (baboon.jpg)

| Model | Skala | PSNR model (dB) | PSNR bicubic (dB) | Waktu (s) |
|---|---:|---:|---:|---:|
| EDSR | ×4 | 20,93 | 20,80 | 14,82 |
| ESPCN | ×4 | 20,62 | 20,80 | 0,06 |
| FSRCNN | ×3 | 21,64 | 21,78 | 0,26 |
| LapSRN | ×8 | 19,42 | 19,57 | 2,22 |

Komentar: hanya EDSR yang PSNR-nya lebih tinggi dari bicubic, dan di gambar zoom bulu baboon
memang sedikit lebih tajam. Tapi EDSR 250× lebih lambat dari ESPCN. ESPCN dan FSRCNN terlihat hampir
sama dengan bicubic. Pada ×8 (LapSRN) semua hasil buram, karena gambar 64×64 piksel terlalu sedikit
menyimpan detail untuk direkonstruksi. ESPCN cocok untuk video real-time, EDSR untuk kualitas terbaik.

### Exercise 5.21 – 5.22: Gabor filter

- 5.21: grid 6×6 kernel. Makin besar panjang gelombang (λ), garis-garisnya makin lebar (frekuensi
  rendah). Pada λ = 20 dan 24 hanya tersisa satu garis karena lebar Gaussian (σ = 5) tetap.
  Orientasi θ memutar arah garis 30° per kolom.
- 5.22: dengan 6 orientasi, tepi yang miring (daun nanas, paruh toucan) tertangkap sedikit lebih
  terang dan rapat dibanding 4 orientasi. Perbedaannya halus, karena 4 orientasi (setiap 45°) sudah
  mencakup sebagian besar arah tepi.

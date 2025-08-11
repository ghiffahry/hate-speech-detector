# Deteksi Ujaran Kebencian Berbasis Deep Learning dengan Pendekatan Statistik

---

## Deskripsi Proyek

Proyek ini merupakan implementasi model deteksi ujaran kebencian (hate speech detection) yang memanfaatkan teknologi **Deep Learning** berbasis Transformer (IndoBERT) dan dipadukan dengan **pendekatan statistika** yang kuat untuk memberikan prediksi yang lebih transparan dan dapat dipercaya.

Berbeda dengan model deteksi konvensional yang hanya memberikan output biner (ya/tidak), model ini mengeluarkan:

- **Probabilitas** bahwa suatu kalimat mengandung ujaran kebencian  
- **Selang prediksi (confidence interval)** untuk menggambarkan rentang keyakinan terhadap prediksi  
- **Visualisasi dan interpretasi** faktor-faktor penyebab model mengklasifikasikan kalimat sebagai ujaran kebencian, sehingga model bukan sekadar "kotak hitam"  

Tujuan utama adalah membantu pengguna dan moderator membuat keputusan yang lebih bijak dan berbasis risiko nyata, serta meningkatkan literasi digital dan keamanan bermedia sosial.

---

## Latar Belakang

Di era digital saat ini, media sosial menjadi ruang penting untuk berkomunikasi dan bertukar pendapat. Namun, tanpa kontrol yang baik, ujaran kebencian bisa tersebar luas, memicu konflik, dan berpotensi melanggar hukum seperti UU ITE di Indonesia. 

Proyek ini bertujuan memberikan solusi yang tidak hanya mengandalkan AI semata, tapi juga didukung dengan landasan statistika yang kuat sehingga hasil prediksi dapat dipertanggungjawabkan secara ilmiah dan transparan.

---

## Dataset

- Dataset terdiri dari **lebih dari 50.000 komentar** dari video YouTube Gibran Rakabuming tentang bonus demografi.  
- Data berisi komentar dalam bahasa Indonesia dengan variasi konteks dan gaya bahasa yang kaya.  
- Dataset telah melalui proses pembersihan (cleaning) dan pelabelan manual untuk kategori ujaran kebencian dan non-ujaran kebencian.

---

## Metodologi

1. **Preprocessing**  
   - Tokenisasi dan normalisasi teks menggunakan tokenizer IndoBERT  
   - Pembersihan data: menghapus karakter khusus, URL, dan emoticon yang tidak relevan  
   
2. **Finetuning Model IndoBERT**  
   - Melakukan pelatihan ulang model IndoBERT pada dataset spesifik ini  
   - Mengoptimasi hyperparameter dengan teknik cross-validation  
   
3. **Pendekatan Statistik**  
   - Menghitung probabilitas prediksi menggunakan output softmax model  
   - Mengestimasi **selang prediksi (confidence interval)** berdasarkan bootstrap sampling atau teknik Monte Carlo Dropout  
   - Menghasilkan interpretasi model menggunakan metode explainability seperti LIME atau SHAP untuk visualisasi fitur penting  

4. **Implementasi API dan Frontend**  
   - API dibuat menggunakan **FastAPI** untuk performa dan kemudahan deployment  
   - Containerized menggunakan **Docker** agar mudah di-deploy di server dengan GPU dan CUDA  
   - Frontend interaktif menggunakan HTML, CSS, dan JavaScript untuk user-friendly interface  

---

## Fitur Utama

- Deteksi probabilistik ujaran kebencian, bukan hanya label ya/tidak  
- Selang kepercayaan prediksi untuk memperlihatkan ketidakpastian model  
- Interpretasi hasil prediksi yang transparan dan mudah dipahami  
- Fitur pemrosesan:  
  - Proses satu kalimat langsung  
  - Proses batch (banyak kalimat sekaligus)  
  - Upload file CSV berisi komentar untuk proses massal  
- Backend API ringan dan cepat, cocok untuk integrasi di berbagai platform  
- Frontend yang mudah digunakan oleh semua kalangan, bukan hanya ahli data  

---

## Evaluasi Model

Model dievaluasi menggunakan metrik-metrik berikut:

| Metrik                | Deskripsi                                       | Hasil (%)     |
|-----------------------|------------------------------------------------|---------------|
| **Akurasi**           | Persentase prediksi benar terhadap total data  | 89.44          |
| **Precision**         | Proporsi prediksi positif yang benar            | 89.50         |
| **Recall (Sensitivity)** | Proporsi data positif yang berhasil terdeteksi | 89.44          |
| **F1-Score**          | Harmonik rata-rata Precision dan Recall         | 89.44          |
| **ROC-AUC**           | Area di bawah kurva ROC (kemampuan klasifikasi) | 96.16          |

> *Catatan:* Model juga diuji dengan **selang prediksi** untuk memastikan prediksi tidak hanya akurat tapi juga dapat dipercaya dengan tingkat kepercayaan tertentu.

---

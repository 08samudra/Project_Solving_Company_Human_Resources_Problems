# 📈 Proyek Data Science: Solusi Permasalahan Human Resources

## 🏢 Business Understanding
PT. Jaya Jaya Maju adalah perusahaan nasional dengan ribuan karyawan yang tersebar di berbagai wilayah Indonesia. Dalam beberapa tahun terakhir, perusahaan mengalami tantangan berupa tingginya tingkat attrition (keluar/resign) karyawan, yang berdampak pada efisiensi operasional, biaya rekrutmen, dan stabilitas tim.

Tim HR ingin:
- Mengetahui faktor-faktor utama yang memengaruhi keputusan karyawan untuk resign
- Memprediksi risiko resign secara otomatis menggunakan machine learning
- Menyediakan dashboard interaktif untuk monitoring dan pengambilan keputusan berbasis data

---

## ❓ Permasalahan Bisnis

1. **Attrition rate** yang tinggi menyebabkan biaya dan beban kerja HR meningkat
2. Belum ada pemetaan faktor utama penyebab resign
3. Tidak tersedia sistem prediksi dan visualisasi data HR yang mudah digunakan

---

## 📌 Ruang Lingkup Proyek

- Eksplorasi dan analisis data karyawan secara menyeluruh
- Preprocessing data: penanganan missing value, encoding, dan pembersihan fitur
- Pembangunan model prediksi attrition menggunakan Logistic Regression
- Pembuatan dashboard Streamlit untuk visualisasi dan prediksi
- Penyusunan insight dan rekomendasi berbasis data untuk HR

---

## 🗂️ Sumber Data & Fitur

- Dataset: [Dicoding Employee Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

    Berikut penjelasan masing-masing variabel pada dataset:

| Nama Variabel           | Tipe         | Penjelasan                                                                 |
|------------------------ |------------- |-------------------------------------------------------------------------- |
| Age                    | Numerik      | Usia karyawan (tahun)                                                     |
| Attrition              | Kategorikal  | Status keluar (1 = keluar, 0 = bertahan)                                  |
| BusinessTravel         | Kategorikal  | Frekuensi perjalanan bisnis (contoh: Non-Travel, Travel_Rarely, dsb)      |
| DailyRate              | Numerik      | Gaji harian karyawan                                                      |
| Department             | Kategorikal  | Departemen tempat karyawan bekerja (contoh: Sales, R&D, HR)               |
| DistanceFromHome       | Numerik      | Jarak rumah ke kantor (dalam satuan mil/km)                               |
| Education              | Ordinal      | Tingkat pendidikan (1 = terendah, 5 = tertinggi)                          |
| EducationField         | Kategorikal  | Bidang pendidikan terakhir (contoh: Life Sciences, Marketing, dsb)        |
| EmployeeCount          | Numerik      | Jumlah karyawan (selalu 1, tidak digunakan)                               |
| EmployeeID             | Numerik      | ID unik karyawan (tidak digunakan untuk analisis)                         |
| EnvironmentSatisfaction| Ordinal      | Tingkat kepuasan terhadap lingkungan kerja (1 = rendah, 4 = sangat puas)  |
| Gender                 | Kategorikal  | Jenis kelamin (Male/Female)                                               |
| HourlyRate             | Numerik      | Gaji per jam                                                              |
| JobInvolvement         | Ordinal      | Tingkat keterlibatan dalam pekerjaan (1 = rendah, 4 = sangat tinggi)      |
| JobLevel               | Ordinal      | Level jabatan (1 = terendah, 5 = tertinggi)                               |
| JobRole                | Kategorikal  | Jabatan spesifik (contoh: Sales Executive, Research Scientist, dsb)       |
| JobSatisfaction        | Ordinal      | Tingkat kepuasan terhadap pekerjaan (1 = rendah, 4 = sangat puas)         |
| MaritalStatus          | Kategorikal  | Status pernikahan (Single, Married, Divorced)                             |
| MonthlyIncome          | Numerik      | Pendapatan bulanan                                                        |
| MonthlyRate            | Numerik      | Gaji bulanan (angka lain, tidak selalu sama dengan MonthlyIncome)         |
| NumCompaniesWorked     | Numerik      | Jumlah perusahaan tempat bekerja sebelumnya                               |
| Over18                 | Kategorikal  | Apakah usia di atas 18 tahun (selalu 'Y', tidak digunakan)                |
| OverTime               | Kategorikal  | Apakah sering lembur (Yes/No)                                             |
| PercentSalaryHike      | Numerik      | Persentase kenaikan gaji terakhir                                         |
| PerformanceRating      | Ordinal      | Penilaian kinerja (1 = terendah, 4 = tertinggi)                           |
| RelationshipSatisfaction| Ordinal     | Kepuasan hubungan kerja (1 = rendah, 4 = sangat puas)                     |
| StandardHours          | Numerik      | Jam kerja standar (selalu 80, tidak digunakan)                            |
| StockOptionLevel       | Ordinal      | Level opsi saham (0 = tidak ada, 3 = tertinggi)                           |
| TotalWorkingYears      | Numerik      | Total tahun pengalaman kerja                                              |
| TrainingTimesLastYear  | Numerik      | Jumlah pelatihan yang diikuti tahun lalu                                  |
| WorkLifeBalance        | Ordinal      | Keseimbangan kerja-hidup (1 = buruk, 4 = sangat baik)                     |
| YearsAtCompany         | Numerik      | Lama bekerja di perusahaan saat ini (tahun)                               |
| YearsInCurrentRole     | Numerik      | Lama bekerja di posisi saat ini (tahun)                                   |
| YearsSinceLastPromotion| Numerik      | Lama sejak promosi terakhir (tahun)                                       |
| YearsWithCurrManager   | Numerik      | Lama bekerja dengan atasan saat ini (tahun)                               |

    > Catatan: Beberapa variabel seperti EmployeeCount, EmployeeID, Over18, StandardHours tidak digunakan dalam analisis/modeling karena tidak informatif atau nilainya konstan.

- Fitur numerik: Age, DailyRate, DistanceFromHome, MonthlyIncome, dsb
- Fitur kategorikal: Department, EducationField, JobRole, MaritalStatus, OverTime, dsb
- Fitur ordinal: Education, JobLevel, EnvironmentSatisfaction, dsb
- Fitur yang di-drop: EmployeeId, EmployeeCount, Over18, StandardHours

---

## 🔧 Setup Environment (VS Code)

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek ini secara lokal di VS Code.

### 1. Kloning Repositori

Buka Terminal pada PC atau Laptop (Windows Win + R, ketik "cmd" dan tekan Enter) (MacOS tekan Command + Space, ketik "Terminal", dan tekan Enter) lalu jalankan:

```bash
git clone https://github.com/08samudra/Project_Solving_Company_Human_Resources_Problems.git
```

Setelah proses cloning selesai, buka folder proyek secara otomatis di VS Code dengan perintah berikut:

```bash
code Project_Solving_Company_Human_Resources_Problems
```

Setelah itu tekan CTRL+` untuk membuka Terminal pada folder proyek di dalam VS Code.

---

### 2. Buat dan Aktifkan Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Jika muncul masalah terkait *Execution Policy*, jalankan PowerShell sebagai Administrator lalu ketik:

```powershell
Set-ExecutionPolicy RemoteSigned
```

Lalu tekan `Y`, dan coba aktivasi ulang.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Setelah aktivasi berhasil, Terminal Anda akan menampilkan `(.venv)` di awal baris.

---

### 3. Install Library yang Dibutuhkan

Pastikan virtual environment sudah aktif. Kemudian install seluruh library yang dibutuhkan melalui berkas `requirements.txt`:

```bash
pip install -r requirements.txt
```
Dan tunggu hingga seluruh proses selesai.

---

## 🚀 Jalankan Aplikasi Streamlit

Membuka dan menjalankan Terminal VS Code.

Lalu jalankan aplikasi Streamlit:

```bash
streamlit run dashboard.py
```

Aplikasi akan terbuka otomatis di browser pada alamat: [http://localhost:8501](http://localhost:8501)

---

## 📊 Business Dashboard

Dashboard Streamlit menyediakan menu:
- **Dashboard Utama**: 
Dashboard ini menyajikan gambaran umum kondisi karyawan di perusahaan, termasuk proporsi attrition (keluar), distribusi usia, pendapatan, serta faktor risiko utama. Visualisasi dan insight di bawah ini membantu HR memahami pola dan kelompok karyawan yang perlu perhatian lebih untuk mencegah attrition.

- **Analisis Attrition**: Pada menu ini, Anda dapat melihat visualisasi dan statistik rata-rata attrition berdasarkan beberapa faktor utama: 
    1. Department: Menampilkan perbandingan tingkat attrition di setiap departemen.
    2. Education Field: Menunjukkan bidang pendidikan mana yang memiliki risiko attrition lebih tinggi.
    3. Job Level: Memvisualisasikan hubungan antara level jabatan dan tingkat attrition.
    4. Marital Status: Menampilkan perbedaan attrition berdasarkan status pernikahan karyawan.

    Setiap grafik dilengkapi dengan nilai persentase attrition untuk memudahkan interpretasi dan pengambilan keputusan oleh tim HR.

- **Prediksi Karyawan**: Form input prediksi risiko resign
Pada menu ini, Anda dapat melakukan simulasi prediksi risiko attrition (keluar) untuk seorang karyawan berdasarkan data individual. Masukkan data karyawan pada form di bawah, lalu sistem akan memproses dan menampilkan prediksi status serta probabilitas risiko keluar. Fitur ini membantu HR dalam melakukan deteksi dini dan intervensi pada karyawan yang berpotensi keluar.

- **Tentang Proyek**: Penjelasan singkat proyek

---

## 🔎 Insight

### Attrition Rate per Department
- Human Resources: 15.79%
- Research & Development: 15.26%
- Sales: 20.69%

### Attrition Rate per Education Field
- Human Resources: 13.33%
- Life Sciences: 16.06%
- Marketing: 21.31%
- Medical: 13.94%
- Other: 16.95%
- Technical Degree: 26.04%

### Attrition Rate per Job Level
- Level 1: 27.41%
- Level 2: 10.16%
- Level 3: 15.15%
- Level 4: 5.00%
- Level 5: 9.09%

### Attrition Rate per Marital Status
- Divorced: 9.50%
- Married: 13.36%
- Single: 26.70%

---

## ✅ Kesimpulan & Rekomendasi

Dashboard interaktif yang dibangun dalam proyek ini terdiri dari beberapa fitur utama:
- **Dashboard Utama:** Menampilkan statistik dan visualisasi komprehensif mengenai kondisi karyawan, proporsi attrition, distribusi usia, pendapatan, serta insight faktor risiko utama secara ringkas.
- **Analisis Attrition:** Menyediakan analisis mendalam dan visualisasi interaktif untuk faktor-faktor utama penyebab attrition, seperti Department, Education Field, Job Level, dan Marital Status, lengkap dengan persentase attrition tiap kategori.
- **Prediksi Karyawan:** Fitur simulasi prediksi risiko resign untuk karyawan individual, sehingga HR dapat melakukan deteksi dini dan intervensi secara personal.

Dashboard ini memudahkan tim HR untuk:
- Memantau kondisi dan tren attrition secara real-time.
- Mengidentifikasi kelompok karyawan yang berisiko tinggi keluar.
- Melakukan prediksi dan intervensi berbasis data untuk meningkatkan retensi karyawan.

- Model prediksi attrition berbasis Logistic Regression mampu mengidentifikasi karyawan berisiko resign dengan akurasi 85%.
- Faktor utama yang memengaruhi attrition adalah Job Level, Marital Status, dan Education Field.
- HR disarankan untuk fokus pada karyawan level 1 dan status single, serta bidang pendidikan tertentu yang memiliki risiko tinggi.
- Dashboard interaktif memudahkan monitoring dan pengambilan keputusan berbasis data.

---

Dibuat oleh: Yoga Samudra

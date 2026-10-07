import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Exploratory Data Analysis (EDA) Dataset Ulasan Mobile JKN
**Penelitian Teknik Informatika - Analisis Sentimen Mobile JKN**

*Notebook ini berisi analisis eksplorasi data komprehensif (Tahap 1 s.d. Tahap 15) terhadap dataset ulasan pengguna aplikasi Mobile JKN dari Google Play Store.*
"""))

# Tahap 1
cells.append(nbf.v4.new_markdown_cell("""## Tahap 1 — Validasi Dataset"""))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATASET_PATH = "../dataset/review_mobile_jkn_raw.csv"

df = pd.read_csv(DATASET_PATH)

print("=== INFORMASI DATASET ===")
print(f"Jumlah baris  : {len(df):,}")
print(f"Jumlah kolom  : {df.shape[1]}")
print(f"Nama kolom    : {df.columns.tolist()}")
print(f"Tipe data     :\n{df.dtypes}")
print(f"Memory Usage  : {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

display(df.head(5))
display(df.tail(5))
"""))

# Tahap 2
cells.append(nbf.v4.new_markdown_cell("""## Tahap 2 — Kualitas Data"""))
cells.append(nbf.v4.new_code_cell("""quality_records = []
for col in df.columns:
    missing_cnt = df[col].isnull().sum()
    missing_pct = (missing_cnt / len(df)) * 100
    unique_cnt = df[col].nunique(dropna=False)
    quality_records.append({
        "Kolom": col,
        "Tipe": str(df[col].dtype),
        "Missing": missing_cnt,
        "Missing %": f"{missing_pct:.2f}%",
        "Unique": unique_cnt
    })

quality_df = pd.DataFrame(quality_records)
display(quality_df)
"""))

# Tahap 3
cells.append(nbf.v4.new_markdown_cell("""## Tahap 3 — Duplikat"""))
cells.append(nbf.v4.new_code_cell("""dup_review_id = df.duplicated(subset=['review_id']).sum()
dup_ulasan = df.duplicated(subset=['ulasan']).sum()
dup_ulasan_rating = df.duplicated(subset=['ulasan', 'rating']).sum()
dup_full = df.duplicated().sum()

dup_df = pd.DataFrame([
    {"Kategori Duplikat": "review_id", "Jumlah": dup_review_id, "Persentase": f"{(dup_review_id/len(df))*100:.2f}%"},
    {"Kategori Duplikat": "ulasan", "Jumlah": dup_ulasan, "Persentase": f"{(dup_ulasan/len(df))*100:.2f}%"},
    {"Kategori Duplikat": "ulasan + rating", "Jumlah": dup_ulasan_rating, "Persentase": f"{(dup_ulasan_rating/len(df))*100:.2f}%"},
    {"Kategori Duplikat": "seluruh baris", "Jumlah": dup_full, "Persentase": f"{(dup_full/len(df))*100:.2f}%"}
])

display(dup_df)
print("Catatan: Tidak ada data duplikat yang dihapus sesuai aturan penting penelitian.")
"""))

# Tahap 4
cells.append(nbf.v4.new_markdown_cell("""## Tahap 4 — Analisis Rating"""))
cells.append(nbf.v4.new_code_cell("""rating_counts = df['rating'].value_counts().sort_index()
rating_df = pd.DataFrame({
    "Rating": rating_counts.index,
    "Jumlah": rating_counts.values,
    "Persentase": [f"{(v/len(df))*100:.2f}%" for v in rating_counts.values]
})
display(rating_df)

plt.figure(figsize=(8, 5))
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
bars = plt.bar(rating_df["Rating"], rating_df["Jumlah"], color=colors, edgecolor="black", alpha=0.85)
plt.title("Distribusi Rating Ulasan Mobile JKN", fontsize=14, fontweight="bold")
plt.xlabel("Rating")
plt.ylabel("Jumlah Ulasan")
for bar in bars:
    height = bar.get_height()
    pct = (height / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 200, f"{height:,}\\n({pct:.1f}%)", ha='center', va='bottom')
plt.show()
"""))

# Tahap 5
cells.append(nbf.v4.new_markdown_cell("""## Tahap 5 — Simulasi Labeling"""))
cells.append(nbf.v4.new_code_cell("""def label_sim_A(r):
    if r in [1, 2]: return "Negatif"
    elif r == 3: return "Netral"
    else: return "Positif"

def label_sim_B(r):
    if r in [1, 2]: return "Negatif"
    else: return "Positif"

df['sim_A'] = df['rating'].apply(label_sim_A)
df['sim_B'] = df['rating'].apply(label_sim_B)

cA = df['sim_A'].value_counts()
cB = df['sim_B'].value_counts()

sim_df = pd.DataFrame([
    {"Skema": "A - 3 kelas", "Negatif": f"{cA.get('Negatif',0):,} ({(cA.get('Negatif',0)/len(df))*100:.2f}%)", "Netral": f"{cA.get('Netral',0):,} ({(cA.get('Netral',0)/len(df))*100:.2f}%)", "Positif": f"{cA.get('Positif',0):,} ({(cA.get('Positif',0)/len(df))*100:.2f}%)"},
    {"Skema": "B - 2 kelas", "Negatif": f"{cB.get('Negatif',0):,} ({(cB.get('Negatif',0)/len(df))*100:.2f}%)", "Netral": "N/A", "Positif": f"{cB.get('Positif',0):,} ({(cB.get('Positif',0)/len(df))*100:.2f}%)"}
])
display(sim_df)
"""))

# Tahap 6
cells.append(nbf.v4.new_markdown_cell("""## Tahap 6 — Analisis Teks"""))
cells.append(nbf.v4.new_code_cell("""ulasan_str = df['ulasan'].fillna("")
char_lens = ulasan_str.apply(len)
word_counts = ulasan_str.apply(lambda x: len(x.split()))

print(f"Ulasan kosong (null/empty)      : {((df['ulasan'].isnull()) | (df['ulasan']=='')).sum()}")
print(f"Ulasan 1 kata                  : {(word_counts == 1).sum()}")
print(f"Ulasan sangat pendek (<3 kata) : {(word_counts < 3).sum()}")
print(f"Ulasan sangat panjang (>50 kata): {(word_counts > 50).sum()}")
print(f"Panjang Karakter (min/max/mean/median): {char_lens.min()} / {char_lens.max()} / {char_lens.mean():.2f} / {char_lens.median()}")
print(f"Panjang Kata (min/max/mean/median)    : {word_counts.min()} / {word_counts.max()} / {word_counts.mean():.2f} / {word_counts.median()}")

plt.figure(figsize=(10, 4))
sns.histplot(word_counts, bins=50, kde=True, color="#3498db")
plt.title("Distribusi Panjang Ulasan (Jumlah Kata)")
plt.xlabel("Jumlah Kata")
plt.ylabel("Frekuensi")
plt.xlim(0, 100)
plt.show()
"""))

# Tahap 7
cells.append(nbf.v4.new_markdown_cell("""## Tahap 7 — Analisis Waktu"""))
cells.append(nbf.v4.new_code_cell("""df['tanggal_dt'] = pd.to_datetime(df['tanggal'], errors='coerce')
df['year_month'] = df['tanggal_dt'].dt.to_period('M')
monthly_counts = df['year_month'].value_counts().sort_index()

print(f"Tanggal terlama : {df['tanggal_dt'].min()}")
print(f"Tanggal terbaru : {df['tanggal_dt'].max()}")

plt.figure(figsize=(12, 4))
monthly_counts.plot(kind='bar', color='#9b59b6', edgecolor='black', alpha=0.85)
plt.title("Jumlah Ulasan per Bulan")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Ulasan")
plt.xticks(rotation=45)
plt.show()
"""))

# Tahap 8
cells.append(nbf.v4.new_markdown_cell("""## Tahap 8 — Analisis Versi Aplikasi"""))
cells.append(nbf.v4.new_code_cell("""ver_counts = df['versi_aplikasi'].value_counts(dropna=False)
top10_ver = ver_counts.head(10)
labels = [str(x) if pd.notnull(x) else "N/A" for x in top10_ver.index]

plt.figure(figsize=(10, 5))
sns.barplot(x=top10_ver.values, y=labels, hue=labels, palette="viridis", legend=False)
plt.title("10 Versi Aplikasi dengan Ulasan Terbanyak")
plt.xlabel("Jumlah Ulasan")
plt.ylabel("Versi Aplikasi")
plt.show()
"""))

# Tahap 9
cells.append(nbf.v4.new_markdown_cell("""## Tahap 9 — Analisis Jumlah Like"""))
cells.append(nbf.v4.new_code_cell("""print(f"Like Minimum : {df['jumlah_like'].min()}")
print(f"Like Maksimum: {df['jumlah_like'].max()}")
print(f"Like Rata-rata: {df['jumlah_like'].mean():.2f}")
print(f"Like Median  : {df['jumlah_like'].median()}")
print(f"Like = 0     : {(df['jumlah_like']==0).sum()} ({(df['jumlah_like']==0).sum()/len(df)*100:.2f}%)")
"""))

# Tahap 10-15 Markdown Summaries
cells.append(nbf.v4.new_markdown_cell("""## Tahap 10 — Menentukan Metode Labeling

| Metode | Kelebihan | Kekurangan | Bias | Beban Kerja | Rekomendasi |
|---|---|---|---|---|---|
| **Rating-based** | 100% konsisten, otomatis, cepat, tidak bergantung annotator human. | Mismatch emosi (sarkasme/noise rating), rating 3 ambigu. | Rendah (objektif dari rating user). | Sangat Rendah. | **Rekomendasi Utama (Skema 2 Kelas)** |
| **Manual Labeling** | Akurasi kontekstual tinggi. | Subjektif, rawan fatigue manusia, butuh waktu lama untuk 21k data. | Tinggi (subjektivitas annotator). | Sangat Tinggi. | Tidak Efektif untuk 21k data |
| **Hybrid Labeling** | Mengombinasikan kecepatan rating & validasi sampel manual. | Membutuhkan pedoman anotasi (annotation guide) dan Inter-Annotator Agreement (Cohen's Kappa). | Sedang. | Sedang. | Alternatif Validasi |

**Rekomendasi**: Rating-based Labeling (Skema B: 2 Kelas - Rating 1-2 Negatif, Rating 4-5 Positif, membuang Rating 3 atau memasukkan rating 3-5 ke Positif).
"""))

cells.append(nbf.v4.new_markdown_cell("""## Tahap 11 — Kelayakan Model SVM & IndoBERT

| Aspek | SVM + TF-IDF | IndoBERT (`indobert-base-p1` / `p2`) |
|---|---|---|
| **Jenis Input** | Sparse Matrix (Bag-of-Words / N-gram) | Dense Token Embeddings (Contextual Sequence) |
| **Representasi Teks** | Frekuensi Kata (TF-IDF) | Self-Attention Mechanism (Contextual Embeddings) |
| **Kelebihan** | Sangat cepat, efisien memory, baseline kuat, interpretable. | Memahami konteks bahasa Indonesia, gaul, tipografik, dan struktur kalimat kompleks. |
| **Kekurangan** | Mengabaikan urutan kata & konteks semantik. | Kebutuhan GPU/RAM tinggi, waktu training lebih lama. |
| **Kebutuhan Komputasi** | Sangat Rendah (Cukup CPU biasa) | Sedang - Tinggi (Membutuhkan GPU seperti Google Colab T4/P100) |
| **Cocok untuk Dataset Ini?** | **SANGAT COCOK** (Baseline Tradisional) | **SANGAT COCOK** (State-of-the-Art Deep Learning) |
"""))

cells.append(nbf.v4.new_markdown_cell("""## Tahap 12 & 13 — Rekomendasi Topik & Judul Penelitian TOP 3

### TOP 1 (Rekomendasi Utama):
**"Analisis Sentimen Ulasan Pengguna Aplikasi Mobile JKN Menggunakan SVM dan IndoBERT Berdasarkan Rating-Based Labeling"**

### TOP 2:
**"Perbandingan Performa dan Efisiensi Komputasi Model SVM dan Fine-Tuned IndoBERT pada Klasifikasi Sentimen Ulasan Mobile JKN"**

### TOP 3:
**"Analisis Sentimen dan Identifikasi Keluhan Utama Pengguna Aplikasi Mobile JKN Berdasarkan Versi Aplikasi Menggunakan IndoBERT"**
"""))

nb['cells'] = cells

with open('notebooks/01_eda_mobile_jkn.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook notebooks/01_eda_mobile_jkn.ipynb successfully created!")

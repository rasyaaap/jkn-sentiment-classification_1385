import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

DATASET_PATH = "dataset/review_mobile_jkn_raw.csv"
OUTPUT_DIR = "results/eda"
SCRATCH_DIR = "scratch"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
num_rows, num_cols = df.shape
col_names = df.columns.tolist()

# Tahap 1
df['tanggal_dt'] = pd.to_datetime(df['tanggal'], errors='coerce')
min_date = df['tanggal_dt'].min()
max_date = df['tanggal_dt'].max()
period_str = f"{min_date.strftime('%Y-%m-%d')} s.d. {max_date.strftime('%Y-%m-%d')}"

head_5_records = df.head(5).copy()
head_5_records['tanggal_dt'] = head_5_records['tanggal_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
tail_5_records = df.tail(5).copy()
tail_5_records['tanggal_dt'] = tail_5_records['tanggal_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

tahap1 = {
    "num_rows": num_rows,
    "num_cols": num_cols,
    "col_names": col_names,
    "mem_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    "period_str": period_str,
    "head_5": head_5_records.to_dict(orient='records'),
    "tail_5": tail_5_records.to_dict(orient='records')
}


# Tahap 2
tahap2_records = []
for col in col_names:
    missing_cnt = int(df[col].isnull().sum())
    missing_pct = float((missing_cnt / num_rows) * 100)
    unique_cnt = int(df[col].nunique(dropna=False))
    dtype_name = str(df[col].dtype)
    tahap2_records.append({
        "Kolom": col,
        "Tipe": dtype_name,
        "Missing": missing_cnt,
        "Missing %": round(missing_pct, 2),
        "Unique": unique_cnt
    })

# Tahap 3
dup_stats = {
    "review_id": {"count": int(df.duplicated(subset=['review_id']).sum()), "pct": round(float(df.duplicated(subset=['review_id']).sum()/num_rows*100), 2)},
    "ulasan": {"count": int(df.duplicated(subset=['ulasan']).sum()), "pct": round(float(df.duplicated(subset=['ulasan']).sum()/num_rows*100), 2)},
    "ulasan_rating": {"count": int(df.duplicated(subset=['ulasan', 'rating']).sum()), "pct": round(float(df.duplicated(subset=['ulasan', 'rating']).sum()/num_rows*100), 2)},
    "full_row": {"count": int(df.duplicated().sum()), "pct": round(float(df.duplicated().sum()/num_rows*100), 2)}
}

# Tahap 4
rating_counts = df['rating'].value_counts().sort_index()
rating_table = []
for r in range(1, 6):
    cnt = int(rating_counts.get(r, 0))
    pct = round(float(cnt / num_rows * 100), 2)
    rating_table.append({"Rating": r, "Jumlah": cnt, "Persentase": pct})

# Figure 1: rating_distribution.png
plt.figure(figsize=(8, 5))
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
bars = plt.bar([x["Rating"] for x in rating_table], [x["Jumlah"] for x in rating_table], color=colors, edgecolor="black", alpha=0.85)
plt.title("Distribusi Rating Ulasan Mobile JKN", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Rating", fontsize=12, labelpad=10)
plt.ylabel("Jumlah Ulasan", fontsize=12, labelpad=10)
plt.xticks([1, 2, 3, 4, 5])
for bar in bars:
    height = bar.get_height()
    pct = (height / num_rows) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 200, f"{height:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"), dpi=300)
plt.close()

# Tahap 5
df['label_sim_A'] = df['rating'].apply(lambda r: "Negatif" if r in [1,2] else ("Netral" if r == 3 else "Positif"))
df['label_sim_B'] = df['rating'].apply(lambda r: "Negatif" if r in [1,2] else "Positif")
cnt_A = df['label_sim_A'].value_counts().to_dict()
cnt_B = df['label_sim_B'].value_counts().to_dict()

simul_table = [
    {
        "Skema": "A - 3 kelas",
        "Negatif": cnt_A.get("Negatif", 0), "Negatif_%": round(cnt_A.get("Negatif", 0)/num_rows*100, 2),
        "Netral": cnt_A.get("Netral", 0), "Netral_%": round(cnt_A.get("Netral", 0)/num_rows*100, 2),
        "Positif": cnt_A.get("Positif", 0), "Positif_%": round(cnt_A.get("Positif", 0)/num_rows*100, 2)
    },
    {
        "Skema": "B - 2 kelas",
        "Negatif": cnt_B.get("Negatif", 0), "Negatif_%": round(cnt_B.get("Negatif", 0)/num_rows*100, 2),
        "Netral": 0, "Netral_%": 0.0,
        "Positif": cnt_B.get("Positif", 0), "Positif_%": round(cnt_B.get("Positif", 0)/num_rows*100, 2)
    }
]

# Tahap 6
ulasan_series = df['ulasan'].fillna("")
empty_cnt = int(((df['ulasan'].isnull()) | (df['ulasan'] == "")).sum())
whitespace_cnt = int(ulasan_series.apply(lambda x: x.strip() == "" and len(x) > 0).sum())
char_lens = ulasan_series.apply(len)
word_counts = ulasan_series.apply(lambda x: len(x.split()))

text_stats = {
    "empty_cnt": empty_cnt,
    "whitespace_cnt": whitespace_cnt,
    "one_word_cnt": int((word_counts == 1).sum()),
    "very_short_words_cnt": int((word_counts < 3).sum()),
    "very_long_words_cnt": int((word_counts > 50).sum()),
    "char_len": {
        "min": int(char_lens.min()),
        "max": int(char_lens.max()),
        "mean": round(float(char_lens.mean()), 2),
        "median": float(char_lens.median())
    },
    "word_len": {
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
        "mean": round(float(word_counts.mean()), 2),
        "median": float(word_counts.median())
    }
}

# Figure 2: review_length_distribution.png
plt.figure(figsize=(10, 5))
sns.histplot(word_counts, bins=50, kde=True, color="#3498db")
plt.title("Distribusi Panjang Ulasan (Jumlah Kata)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Jumlah Kata per Ulasan", fontsize=12, labelpad=10)
plt.ylabel("Frekuensi", fontsize=12, labelpad=10)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "review_length_distribution.png"), dpi=300)
plt.close()

# Tahap 7
df['year_month'] = df['tanggal_dt'].dt.to_period('M')
monthly_series = df['year_month'].value_counts().sort_index()
monthly_data = {str(k): int(v) for k, v in monthly_series.items()}

# Figure 3: monthly_reviews.png
plt.figure(figsize=(12, 5))
monthly_series.plot(kind='bar', color='#9b59b6', edgecolor='black', alpha=0.85)
plt.title("Jumlah Ulasan Mobile JKN per Bulan", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Bulan (Tahun-Bulan)", fontsize=12, labelpad=10)
plt.ylabel("Jumlah Ulasan", fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_reviews.png"), dpi=300)
plt.close()

# Tahap 8
num_versions = int(df['versi_aplikasi'].nunique(dropna=True))
null_versions = int(df['versi_aplikasi'].isnull().sum())
version_counts = df['versi_aplikasi'].value_counts(dropna=False)
top20_versions = []
for ver, cnt in version_counts.head(20).items():
    ver_str = str(ver) if pd.notnull(ver) else "Tidak Diketahui (NaN)"
    top20_versions.append({"versi": ver_str, "jumlah": int(cnt), "persentase": round(float(cnt/num_rows*100), 2)})

# Figure 4: version_distribution.png
plt.figure(figsize=(12, 6))
top10_ver = version_counts.head(10)
labels = [str(x) if pd.notnull(x) else "N/A" for x in top10_ver.index]
sns.barplot(x=top10_ver.values, y=labels, hue=labels, palette="viridis", legend=False)
plt.title("10 Versi Aplikasi Mobile JKN dengan Ulasan Terbanyak", fontsize=14, fontweight="bold", pad=15)

plt.xlabel("Jumlah Ulasan", fontsize=12, labelpad=10)
plt.ylabel("Versi Aplikasi", fontsize=12, labelpad=10)
for idx, val in enumerate(top10_ver.values):
    pct = (val / num_rows) * 100
    plt.text(val + 100, idx, f"{val:,} ({pct:.1f}%)", va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "version_distribution.png"), dpi=300)
plt.close()

# Tahap 9
like_stats = {
    "min": int(df['jumlah_like'].min()),
    "max": int(df['jumlah_like'].max()),
    "mean": round(float(df['jumlah_like'].mean()), 2),
    "median": float(df['jumlah_like'].median()),
    "like_zero_cnt": int((df['jumlah_like'] == 0).sum()),
    "like_zero_pct": round(float((df['jumlah_like'] == 0).sum() / num_rows * 100), 2)
}

# Combine into master json
eda_results = {
    "tahap1": tahap1,
    "tahap2": tahap2_records,
    "tahap3": dup_stats,
    "tahap4": rating_table,
    "tahap5": simul_table,
    "tahap6": text_stats,
    "tahap7": {"min_date": min_date.strftime('%Y-%m-%d'), "max_date": max_date.strftime('%Y-%m-%d'), "monthly": monthly_data},
    "tahap8": {"num_versions": num_versions, "null_versions": null_versions, "top20": top20_versions},
    "tahap9": like_stats
}

with open(os.path.join(SCRATCH_DIR, "eda_results.json"), "w", encoding="utf-8") as f:
    json.dump(eda_results, f, ensure_ascii=False, indent=2)

print("EDA output and figures successfully generated and saved!")

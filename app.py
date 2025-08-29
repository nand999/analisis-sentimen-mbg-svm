# ============================================================
# Aplikasi Streamlit Analisis Sentimen Bahasa Indonesia
# Preprocessing (cleaning -> casefolding -> tokenizing -> stopwords -> normalisasi -> stemming)
# TF-IDF + SVM (LinearSVC) dengan balancing (oversampling)
# Visualisasi: Pie chart, WordCloud per kelas, Top kata per kelas
# Tabel tahap preprocessing
# Input teks untuk prediksi cepat (pipeline disimpan & dimuat)
# ============================================================

# Cara menjalankan:
# 1) pip install streamlit pandas scikit-learn Sastrawi nltk wordcloud plotly joblib matplotlib
# 2) streamlit run app.py
# 3) Pastikan file mbg.csv tersedia dengan kolom: 'text' (str) dan 'sentimen' (0/1)

import os
import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from joblib import dump, load

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# -------------------------
# Konfigurasi halaman
# -------------------------
st.set_page_config(page_title="Analisis Sentimen - Bahasa Indonesia (SVM + TF-IDF)", layout="wide")

# -------------------------
# Utilities: resources
# -------------------------
@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def get_stopwords_id() -> set:
    # Pastikan stopwords tersedia
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    sw = set()
    try:
        sw.update(nltk_stopwords.words('indonesian'))
    except Exception:
        pass
    # Tambahkan stopwords umum/slang
    extra = {
        'yg', 'yang', 'dgn', 'dengan', 'aja', 'saja', 'nya', 'nih', 'sih', 'lah', 'kah',
        'banget', 'bgt', 'ga', 'gak', 'nggak', 'enggak', 'kok', 'dong', 'pun', 'dst',
        'dll', 'dkk', 'dr', 'dari', 'utk', 'untuk', 'ke', 'pd', 'pada', 'tp', 'tapi',
        'krn', 'karena', 'kalo', 'kalau', 'lg', 'lagi', 'udah', 'sdh', 'sudah', 'blm',
        'belum', 'bisa', 'bs', 'trs', 'terus', 'aja', 'aja', 'kak', 'bro', 'sis', 'gan',
        'klo', 'kmu', 'kamu', 'sy', 'saya', 'aku', 'ku', 'mu', 'lo', 'lu', 'jd', 'jadi',
        'doang', 'bang', 'mbak', 'mas'
    }
    sw.update(extra)
    return sw

@st.cache_resource
def get_normalization_dict() -> Dict[str, str]:
    # Kamus normalisasi slang/alay sederhana; bisa diperluas
    # Jika Anda punya kamus eksternal, bisa gabungkan di sini.
    return {
        'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','enggak':'tidak','tak':'tidak','tdk':'tidak',
        'bgt':'banget','bgd':'banget','bngt':'banget',
        'yg':'yang','yng':'yang',
        'utk':'untuk','buat':'untuk',
        'sm':'sama','sma':'sama','sama':'dengan',
        'krn':'karena','krna':'karena',
        'dr':'dari','drpd':'daripada','drpda':'daripada',
        'dgn':'dengan','dg':'dengan','dengan':'dengan',
        'pd':'pada','pda':'pada',
        'jg':'juga','jga':'juga',
        'dlm':'dalam','dlam':'dalam',
        'lg':'lagi','lgi':'lagi',
        'udh':'sudah','udah':'sudah','sdh':'sudah',
        'blm':'belum','blom':'belum',
        'bisa':'bisa','bs':'bisa',
        'bkn':'bukan','bukan':'bukan',
        'spt':'seperti','sprti':'seperti',
        'krg':'kurang','kurg':'kurang',
        'tlg':'tolong','plis':'tolong','please':'tolong',
        'mksd':'maksud','mksh':'terima kasih','makasih':'terima kasih','makasi':'terima kasih','thx':'terima kasih',
        'smg':'semoga','semoga':'semoga',
        'dl':'dulu','dlu':'dulu',
        'n':'dan','&':'dan',
        'aja':'saja','doang':'saja','aja':'saja',
        'km':'kamu','kmu':'kamu','loe':'kamu','lo':'kamu','lu':'kamu',
        'sy':'saya','aq':'aku','gw':'saya','gue':'saya','gua':'saya',
        'ok':'oke','oke':'oke','okey':'oke','okeyy':'oke',
        'mnt':'menit','hr':'hari','bln':'bulan','thn':'tahun',
        'td':'tadi','tadi':'tadi',
        'tp':'tapi','tpi':'tapi'
    }

# -------------------------
# Preprocessing functions
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Hilangkan URL, mention, hashtag, RT
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    # Hilangkan angka dan simbol non-huruf, sisakan spasi dan huruf
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    # Reduksi huruf berulang (cooool -> cool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def casefold_text(text: str) -> str:
    return text.lower()

def tokenize(text: str) -> List[str]:
    # Setelah cleaning, cukup split whitespace
    return text.split()

def remove_stopwords(tokens: List[str], stopwords_set: set) -> List[str]:
    return [t for t in tokens if t not in stopwords_set and len(t) > 1]

def normalize_tokens(tokens: List[str], norm_dict: Dict[str, str]) -> List[str]:
    return [norm_dict.get(t, t) for t in tokens]

def stem_string(text: str, stemmer) -> str:
    # Sastrawi lebih optimal bila diberi string utuh
    return stemmer.stem(text)

def pipeline_steps(text: str, stop_set: set, norm_dict: Dict[str, str], stemmer) -> Dict[str, str]:
    s1 = clean_text(text)
    s2 = casefold_text(s1)
    toks = tokenize(s2)
    s3 = " ".join(toks)
    toks2 = remove_stopwords(toks, stop_set)
    s4 = " ".join(toks2)
    toks3 = normalize_tokens(toks2, norm_dict)
    s5 = " ".join(toks3)
    s6 = stem_string(s5, stemmer)
    # final tokens
    final_tokens = s6.split()
    s6_join = " ".join(final_tokens)
    return {
        "cleaning": s1,
        "casefolding": s2,
        "tokenizing": s3,
        "stopwords_removed": s4,
        "normalisasi": s5,
        "stemming": s6_join
    }

# -------------------------
# Custom Transformer for sklearn Pipeline
# -------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords_set: set = None, norm_dict: Dict[str, str] = None, stemmer=None):
        self.stopwords_set = stopwords_set if stopwords_set is not None else get_stopwords_id()
        self.norm_dict = norm_dict if norm_dict is not None else get_normalization_dict()
        self.stemmer = stemmer if stemmer is not None else get_stemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        processed = []
        for text in X:
            steps = pipeline_steps(text, self.stopwords_set, self.norm_dict, self.stemmer)
            processed.append(steps["stemming"])  # hasil akhir yang bersih
        return processed

# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv_default(path: str = "mbg.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    # Coba beberapa encoding umum
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return None
    # Standarisasi kolom
    if 'text' not in df.columns or 'sentimen' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'text' dan 'sentimen'.")
    # Pastikan label 0/1
    df['sentimen'] = df['sentimen'].astype(int)
    return df

# -------------------------
# Balancing dataset (oversampling per teks)
# -------------------------
def make_balanced(df: pd.DataFrame, label_col: str = "sentimen") -> pd.DataFrame:
    counts = df[label_col].value_counts()
    max_n = counts.max()
    frames = []
    for c, n in counts.items():
        df_c = df[df[label_col] == c]
        if n < max_n:
            df_up = resample(df_c, replace=True, n_samples=max_n, random_state=42)
        else:
            df_up = df_c
        frames.append(df_up)
    df_bal = pd.concat(frames, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df_bal

# -------------------------
# Model build/save/load
# -------------------------
MODEL_PATH = "model_svm_tfidf.joblib"

@st.cache_resource(show_spinner=False)
def build_pipeline(ngram_min=1, ngram_max=2, max_features=30000, C=1.0):
    preproc = TextPreprocessor(
        stopwords_set=get_stopwords_id(),
        norm_dict=get_normalization_dict(),
        stemmer=get_stemmer()
    )
    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: s.split(),
        token_pattern=None,
        lowercase=False,
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        sublinear_tf=True
    )
    clf = LinearSVC(C=C, class_weight='balanced', random_state=42)
    pipe = Pipeline([
        ("prep", preproc),
        ("tfidf", vectorizer),
        ("svm", clf)
    ])
    return pipe

def train_and_save(df: pd.DataFrame, ngram_min=1, ngram_max=2, max_features=30000, C=1.0) -> Pipeline:
    pipe = build_pipeline(ngram_min, ngram_max, max_features, C)
    df_bal = make_balanced(df, label_col="sentimen")
    X_train = df_bal["text"].astype(str).tolist()
    y_train = df_bal["sentimen"].astype(int).values
    pipe.fit(X_train, y_train)
    dump(pipe, MODEL_PATH)
    return pipe

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return load(MODEL_PATH)
        except Exception:
            return None
    return None

# -------------------------
# Visualization helpers
# -------------------------
def plot_sentiment_pie(df: pd.DataFrame):
    label_map = {0: "Negatif", 1: "Positif"}
    counts = df['sentimen'].map(label_map).value_counts().reset_index()
    counts.columns = ["sentimen", "jumlah"]
    fig = px.pie(counts, names="sentimen", values="jumlah", color="sentimen",
                 color_discrete_map={"Negatif": "#EF553B", "Positif": "#00CC96"},
                 hole=0.3, title="Distribusi Sentimen (Data Asli)")
    st.plotly_chart(fig, use_container_width=True)

def generate_wordcloud(texts: List[str], title: str, stopset: set):
    if not texts:
        st.warning(f"Tidak ada teks untuk Word Cloud: {title}")
        return
    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        stopwords=stopset,
        collocations=False
    ).generate(" ".join(texts))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

def top_terms_per_class(texts: List[str], y: np.ndarray, n=20, ngram=(1,1), max_features=50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Menghitung n-gram paling sering per kelas dari teks yang sudah dipreprocess
    cv = CountVectorizer(
        tokenizer=lambda s: s.split(),
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram,
        max_features=max_features
    )
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    y = np.asarray(y)
    pos_idx = (y == 1)
    neg_idx = (y == 0)

    # Hati-hati jika kelas kosong
    if pos_idx.sum() == 0 or neg_idx.sum() == 0:
        return pd.DataFrame(columns=["term", "count"]), pd.DataFrame(columns=["term", "count"])

    pos_counts = np.asarray(X[pos_idx].sum(axis=0)).ravel()
    neg_counts = np.asarray(X[neg_idx].sum(axis=0)).ravel()

    pos_top_idx = np.argsort(pos_counts)[::-1][:n]
    neg_top_idx = np.argsort(neg_counts)[::-1][:n]

    df_pos = pd.DataFrame({"term": vocab[pos_top_idx], "count": pos_counts[pos_top_idx]})
    df_neg = pd.DataFrame({"term": vocab[neg_top_idx], "count": neg_counts[neg_top_idx]})
    return df_pos, df_neg

def bar_top_terms(df_terms: pd.DataFrame, title: str, color="#636EFA"):
    if df_terms.empty:
        st.info(f"Tidak ada data untuk {title}")
        return
    fig = px.bar(df_terms.sort_values("count", ascending=True), x="count", y="term", orientation="h",
                 title=title, color_discrete_sequence=[color])
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# APP UI
# -------------------------
st.title("Analisis Sentimen Bahasa Indonesia (SVM + TF-IDF)")
st.caption("Preprocessing lengkap • TF-IDF • SVM • Balancing untuk model • Dashboard interaktif")

# Sidebar: parameter dan aksi
st.sidebar.header("Pengaturan Model")
ng_min = st.sidebar.number_input("N-gram min", min_value=1, max_value=3, value=1, step=1)
ng_max = st.sidebar.number_input("N-gram max", min_value=1, max_value=3, value=2, step=1)
if ng_max < ng_min:
    st.sidebar.error("N-gram max harus >= N-gram min")
max_feat = st.sidebar.slider("Max fitur TF-IDF", min_value=1000, max_value=100000, value=30000, step=1000)
C_val = st.sidebar.select_slider("C (LinearSVC)", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)

st.sidebar.header("Model")
btn_train = st.sidebar.button("Latih / Perbarui Model (Balanced)")
btn_load = st.sidebar.button("Muat Model Tersimpan")

# Load data
df = load_csv_default("mbg.csv")
if df is None:
    st.warning("File 'mbg.csv' tidak ditemukan. Silakan unggah file melalui widget di bawah.")
    up = st.file_uploader("Unggah mbg.csv (harus mengandung kolom 'text' dan 'sentimen' [0/1])", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if 'text' not in df.columns or 'sentimen' not in df.columns:
            st.error("File harus memiliki kolom 'text' dan 'sentimen'")
            st.stop()
        df['sentimen'] = df['sentimen'].astype(int)
    else:
        st.stop()

# Tampilkan ringkas data
with st.expander("Lihat sample data"):
    st.dataframe(df.head(20), use_container_width=True)

# Distribusi sentimen (data asli)
plot_sentiment_pie(df)

# Siapkan preprocessor untuk kebutuhan visualisasi (pakai data asli, tidak balancing)
stop_set = get_stopwords_id()
norm_dict = get_normalization_dict()
stemmer = get_stemmer()
preproc = TextPreprocessor(stopwords_set=stop_set, norm_dict=norm_dict, stemmer=stemmer)

# Cache hasil preprocessing untuk dashboard
@st.cache_data(show_spinner=False)
def preprocess_for_dashboard(texts: List[str]) -> List[str]:
    return preproc.transform(texts)

# Preprocess teks untuk visualisasi (data asli)
pre_texts = preprocess_for_dashboard(df["text"].astype(str).tolist())
df_viz = df.copy()
df_viz["text_preprocessed"] = pre_texts

# WordCloud per kelas (dari data asli, setelah preprocessing agar lebih informatif)
col_wc1, col_wc2 = st.columns(2)
with col_wc1:
    st.subheader("Word Cloud - Negatif")
    generate_wordcloud(df_viz.loc[df_viz["sentimen"] == 0, "text_preprocessed"].tolist(),
                       "Negatif", stop_set)
with col_wc2:
    st.subheader("Word Cloud - Positif")
    generate_wordcloud(df_viz.loc[df_viz["sentimen"] == 1, "text_preprocessed"].tolist(),
                       "Positif", stop_set)

# Top kata per kelas
st.subheader("Kata/Frasa Paling Sering per Kelas (Data Asli, setelah preprocessing)")
df_pos, df_neg = top_terms_per_class(
    texts=df_viz["text_preprocessed"].tolist(),
    y=df_viz["sentimen"].values,
    n=20,
    ngram=(1, 2),
    max_features=50000
)
col_top1, col_top2 = st.columns(2)
with col_top1:
    bar_top_terms(df_neg, "Top Terms - Negatif", color="#EF553B")
with col_top2:
    bar_top_terms(df_pos, "Top Terms - Positif", color="#00CC96")

# Tabel tahap preprocessing (contoh beberapa baris)
st.subheader("Tabel Hasil Tahap Preprocessing (Cleaning → Casefolding → Tokenizing → Stopwords → Normalisasi → Stemming)")
n_rows = st.slider("Jumlah baris contoh", 5, 50, 12, 1)
sample_df = df.sample(n=min(n_rows, len(df)), random_state=42).copy().reset_index(drop=True)

def build_preprocess_table(sdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sdf.iterrows():
        steps = pipeline_steps(str(r["text"]), stop_set, norm_dict, stemmer)
        rows.append({
            "text_asli": r["text"],
            "cleaning": steps["cleaning"],
            "casefolding": steps["casefolding"],
            "tokenizing": steps["tokenizing"],
            "stopwords_removed": steps["stopwords_removed"],
            "normalisasi": steps["normalisasi"],
            "stemming": steps["stemming"],
            "sentimen": r["sentimen"]
        })
    return pd.DataFrame(rows)

tbl = build_preprocess_table(sample_df)
st.dataframe(tbl, use_container_width=True, height=350)

st.markdown("---")

# Latih / Muat model
if "model" not in st.session_state:
    st.session_state.model = None

if btn_train:
    with st.spinner("Melatih model pada dataset yang DISEIMBANGKAN (oversampling)..."):
        try:
            model = train_and_save(df, ngram_min=ng_min, ngram_max=ng_max, max_features=max_feat, C=C_val)
            st.session_state.model = model
            st.success(f"Model selesai dilatih dan disimpan ke {MODEL_PATH}")
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")

if btn_load:
    model = load_model()
    if model is None:
        st.error("Model belum tersedia. Silakan latih dulu.")
    else:
        st.session_state.model = model
        st.success(f"Model dimuat dari {MODEL_PATH}")

# Input Prediksi
st.header("Uji Prediksi Teks Baru")
with st.form("pred_form"):
    user_text = st.text_area("Masukkan teks (Bahasa Indonesia)", height=130,
                             placeholder="Contoh: Layanannya cepat dan memuaskan, terima kasih!")
    colf = st.columns(2)
    with colf[0]:
        show_pre = st.checkbox("Tampilkan hasil preprocessing teks input", value=True)
    with colf[1]:
        _ = st.caption("Sentimen: 0=Negatif, 1=Positif")
    submitted = st.form_submit_button("Prediksi Sentimen")

if submitted:
    # Pastikan ada model
    if st.session_state.model is None:
        # Otomatis coba load model yang tersimpan
        mdl = load_model()
        if mdl is not None:
            st.session_state.model = mdl
        else:
            st.warning("Model belum tersedia, melatih model cepat dengan pengaturan saat ini...")
            try:
                mdl = train_and_save(df, ngram_min=ng_min, ngram_max=ng_max, max_features=max_feat, C=C_val)
                st.session_state.model = mdl
                st.success(f"Model dilatih cepat dan disimpan ke {MODEL_PATH}")
            except Exception as e:
                st.error(f"Gagal melatih/memuat model: {e}")
                st.stop()

    model = st.session_state.model
    try:
        pred = model.predict([user_text])[0]
        label = "Positif" if int(pred) == 1 else "Negatif"
        color = "#00CC96" if int(pred) == 1 else "#EF553B"
        st.markdown(f"Prediksi: <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)

        if show_pre:
            steps = pipeline_steps(user_text, stop_set, norm_dict, stemmer)
            st.write("Hasil preprocessing input:")
            st.json(steps)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")

# Footer info
st.markdown("---")
st.caption("Catatan:")
st.markdown("""
- Dashboard menggunakan data asli tanpa balancing (merepresentasikan kondisi sebenarnya).
- Model untuk prediksi dilatih dengan dataset yang sudah di-oversample agar seimbang, sehingga performa lebih stabil pada kelas minor.
- Pipeline sudah mencakup: preprocessing lengkap → TF-IDF → SVM. Proses prediksi input baru berjalan cepat berkat model yang disimpan.
""")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sentiment_model import SentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set font untuk mendukung bahasa Indonesia
plt.rcParams['font.family'] = 'DejaVu Sans'

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Bahasa Indonesia",
    page_icon="📊",
    layout="wide"
)

# Load CSS untuk styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load dataset yang sudah diproses"""
    try:
        df = pd.read_csv('mbg_processed.csv')
        return df
    except FileNotFoundError:
        st.error("File mbg_processed.csv tidak ditemukan. Jalankan sentiment_model.py terlebih dahulu.")
        return None

@st.cache_resource
def load_model():
    """Load model yang sudah dilatih"""
    try:
        analyzer = SentimentAnalyzer()
        analyzer.load_model('sentiment_model.pkl')
        return analyzer
    except FileNotFoundError:
        st.error("Model tidak ditemukan. Jalankan sentiment_model.py terlebih dahulu.")
        return None

def create_pie_chart(df):
    """Membuat pie chart distribusi sentimen"""
    sentiment_counts = df['sentiment'].value_counts()
    labels = ['Negatif', 'Positif']
    values = [sentiment_counts[0], sentiment_counts[1]]
    colors = ['#ff7f7f', '#7fbf7f']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.3,
        marker=dict(colors=colors),
        texttemplate='<b>%{label}</b><br>%{value}<br>(%{percent})'
    )])
    
    fig.update_traces(
        textposition='inside', 
        textfont_size=12
    )
    
    fig.update_layout(
        title={
            'text': "Distribusi Sentimen Dataset",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig

def create_wordcloud(text, title, colormap='viridis'):
    """Membuat word cloud dengan font yang mendukung bahasa Indonesia"""
    if len(text) == 0:
        return None
    
    # Gabungkan semua teks
    combined_text = ' '.join(text)
    
    if not combined_text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10,
        prefer_horizontal=0.9,
        collocations=False
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

def get_top_words(texts, n=20):
    """Mendapatkan kata-kata yang paling sering muncul"""
    all_words = []
    for text in texts:
        if pd.notna(text) and text.strip():
            all_words.extend(text.split())
    
    # Filter kata yang terlalu pendek
    all_words = [word for word in all_words if len(word) > 2]
    
    word_freq = Counter(all_words)
    return word_freq.most_common(n)

def show_preprocessing_steps(analyzer, text):
    """Menampilkan tabel langkah-langkah preprocessing"""
    if text.strip():
        processed_text, steps = analyzer.preprocess_text(text, show_steps=True)
        
        st.subheader("📋 Tahapan Preprocessing")
        
        # Buat DataFrame untuk menampilkan steps
        steps_data = [
            ["🔤 Teks Asli", steps['original']],
            ["🧹 Pembersihan", steps['cleaned']],
            ["📝 Case Folding", steps['casefolded']],
            ["✂️ Tokenisasi", str(steps['tokenized'])],
            ["🚫 Hapus Stopwords", str(steps['no_stopwords'])],
            ["🔄 Normalisasi", str(steps['normalized'])],
            ["🌱 Stemming", str(steps['stemmed'])],
            ["✅ Hasil Akhir", steps['final']]
        ]
        
        steps_df = pd.DataFrame(steps_data, columns=["Tahap", "Hasil"])
        
        # Styling untuk DataFrame
        def highlight_rows(row):
            if row.name == 0:  # Original text
                return ['background-color: #e8f4fd; color: black'] * len(row)
            elif row.name == len(steps_df) - 1:  # Final result
                return ['background-color: #e8f5e8; color: black'] * len(row)
            else:
                return ['background-color: #f9f9f9; color: black'] * len(row)
        
        styled_df = steps_df.style.apply(highlight_rows, axis=1)

        # Tambahkan CSS agar teks di dataframe selalu hitam
        st.markdown(
            """
            <style>
            .stDataFrame div, .stDataFrame table, .stDataFrame th, .stDataFrame td {
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.dataframe(styled_df, use_container_width=True, height=320)
        
        return processed_text
    return None


def show_examples():
    """Menampilkan contoh-contoh teks untuk analisis"""
    st.subheader("💡 Contoh Teks untuk Dicoba")
    
    examples = [
        "Pelayanannya sangat memuaskan dan staffnya ramah sekali!",
        "Makanannya enak banget, pasti bakal balik lagi kesini",
        "Pelayanan buruk banget, lama dan tidak profesional",
        "Harga mahal tapi kualitasnya mengecewakan",
        "Tempat nya nyaman, cocok buat nongkrong sama temen",
        "Antrian panjang banget, capek nunggu",
        "Produknya berkualitas tinggi dengan harga yang terjangkau",
        "Kecewa banget sama pelayanan disini, tidak akan kembali lagi"
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Contoh {i}: {example[:50]}...", key=f"example_{i}"):
            return example
    
    return None

def main():
    """Fungsi utama dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Analisis Sentimen Bahasa Indonesia</h1>', unsafe_allow_html=True)
    
    # Load data dan model
    df = load_data()
    analyzer = load_model()
    
    if df is None or analyzer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🧭 Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["📈 Dashboard Utama", "🔮 Prediksi Sentimen", "📋 Demo Preprocessing"]
    )
    
    if page == "📈 Dashboard Utama":
        show_main_dashboard(df)
    elif page == "🔮 Prediksi Sentimen":
        show_prediction_page(analyzer)
    elif page == "📋 Demo Preprocessing":
        show_preprocessing_demo(analyzer)

def show_main_dashboard(df):
    """Menampilkan dashboard utama"""
    st.header("📊 Dashboard Analisis Sentimen")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_data = len(df)
    positive_count = len(df[df['sentiment'] == 1])
    negative_count = len(df[df['sentiment'] == 0])
    positive_ratio = (positive_count / total_data) * 100
    
    with col1:
        st.metric("📄 Total Data", f"{total_data:,}")
    with col2:
        st.metric("😊 Sentimen Positif", f"{positive_count:,}")
    with col3:
        st.metric("😞 Sentimen Negatif", f"{negative_count:,}")
    with col4:
        st.metric("📊 Rasio Positif", f"{positive_ratio:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart
        fig_pie = create_pie_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar Chart untuk top words secara keseluruhan
        all_texts = df['processed_text'].dropna().tolist()
        top_words = get_top_words(all_texts, 15)
        
        if top_words:
            words, counts = zip(*top_words)
            fig_bar = go.Figure([go.Bar(
                x=list(counts), 
                y=list(words), 
                orientation='h',
                marker_color='lightblue'
            )])
            fig_bar.update_layout(
                title="15 Kata Teratas (Keseluruhan)",
                xaxis_title="Frekuensi",
                yaxis_title="Kata",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Word Clouds
    st.header("☁️ Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Sentimen Positif")
        positive_texts = df[df['sentiment'] == 1]['processed_text'].dropna().tolist()
        if positive_texts:
            fig_wc_pos = create_wordcloud(positive_texts, "Word Cloud Sentimen Positif", 'Greens')
            if fig_wc_pos:
                st.pyplot(fig_wc_pos, clear_figure=True)
        else:
            st.info("Tidak ada data sentimen positif")
    
    with col2:
        st.subheader("😞 Sentimen Negatif")
        negative_texts = df[df['sentiment'] == 0]['processed_text'].dropna().tolist()
        if negative_texts:
            fig_wc_neg = create_wordcloud(negative_texts, "Word Cloud Sentimen Negatif", 'Reds')
            if fig_wc_neg:
                st.pyplot(fig_wc_neg, clear_figure=True)
        else:
            st.info("Tidak ada data sentimen negatif")
    
    # Top words by sentiment
    st.header("📈 Kata-kata Teratas per Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Top 15 Kata Sentimen Positif")
        if positive_texts:
            top_words_pos = get_top_words(positive_texts, 15)
            if top_words_pos:
                words_pos, counts_pos = zip(*top_words_pos)
                
                fig_pos = go.Figure([go.Bar(
                    x=list(counts_pos), 
                    y=list(words_pos), 
                    orientation='h',
                    marker_color='lightgreen'
                )])
                fig_pos.update_layout(
                    xaxis_title="Frekuensi",
                    yaxis_title="Kata",
                    height=400
                )
                st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.subheader("😞 Top 15 Kata Sentimen Negatif")
        if negative_texts:
            top_words_neg = get_top_words(negative_texts, 15)
            if top_words_neg:
                words_neg, counts_neg = zip(*top_words_neg)
                
                fig_neg = go.Figure([go.Bar(
                    x=list(counts_neg), 
                    y=list(words_neg), 
                    orientation='h',
                    marker_color='lightcoral'
                )])
                fig_neg.update_layout(
                    xaxis_title="Frekuensi",
                    yaxis_title="Kata",
                    height=400
                )
                st.plotly_chart(fig_neg, use_container_width=True)
    
    # Data Sample
    st.header("📋 Sample Data")
    sample_df = df[['text', 'sentiment', 'processed_text']].head(10).copy()
    sample_df['sentiment'] = sample_df['sentiment'].map({0: 'Negatif', 1: 'Positif'})
    st.dataframe(sample_df, use_container_width=True)

def show_prediction_page(analyzer):
    """Halaman prediksi sentimen"""
    st.header("🔮 Prediksi Sentimen")
    
    st.write("Masukkan teks bahasa Indonesia untuk memprediksi sentimennya:")
    
    # Contoh teks
    selected_example = show_examples()
    
    # Input teks
    default_text = selected_example if selected_example else ""
    user_input = st.text_area(
        "Teks untuk dianalisis:",
        height=100,
        value=default_text,
        placeholder="Contoh: Pelayanannya sangat memuaskan dan staffnya ramah sekali!"
    )
    
    if st.button("🚀 Analisis Sentimen", type="primary"):
        if user_input.strip():
            with st.spinner("Menganalisis sentimen..."):
                # Prediksi
                result = analyzer.predict_sentiment(user_input)
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hasil prediksi
                    sentiment_color = "green" if result['sentiment'] == "Positif" else "red"
                    sentiment_icon = "😊" if result['sentiment'] == "Positif" else "😞"
                    
                    st.markdown(f"""
                    <div style="padding: 2rem; border: 3px solid {sentiment_color}; border-radius: 1rem; text-align: center; background-color: rgba({'0,255,0' if sentiment_color == 'green' else '255,0,0'}, 0.1);">
                        <h2 style="color: {sentiment_color}; margin: 0;">{sentiment_icon} {result['sentiment']}</h2>
                        <h3 style="margin: 0.5rem 0;">Confidence: {result['confidence']:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Probability chart
                    fig = go.Figure([go.Bar(
                        x=['😞 Negatif', '😊 Positif'],
                        y=[result['probability_negative'], result['probability_positive']],
                        marker_color=['lightcoral', 'lightgreen'],
                        text=[f"{result['probability_negative']:.1%}", f"{result['probability_positive']:.1%}"],
                        textposition='auto'
                    )])
                    fig.update_layout(
                        title="Probabilitas Sentimen",
                        yaxis_title="Probabilitas",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Preprocessing steps
                st.markdown("---")
                show_preprocessing_steps(analyzer, user_input)
                
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

def show_preprocessing_demo(analyzer):
    """Halaman demo preprocessing"""
    st.header("📋 Demo Preprocessing")
    
    st.write("Lihat bagaimana teks bahasa Indonesia diproses melalui setiap tahap preprocessing:")
    
    # Input teks
    demo_text = st.text_area(
        "Masukkan teks untuk melihat proses preprocessing:",
        height=100,
        placeholder="Contoh: Pelayanannya bgt bagus dan staffnya ramah bgt!!! Recommended deh 😊",
        value="Pelayanannya bgt bagus dan staffnya ramah bgt!!! Recommended deh 😊"
    )
    
    if demo_text.strip():
        show_preprocessing_steps(analyzer, demo_text)
        
        # Penjelasan setiap tahap
        st.subheader("📚 Penjelasan Tahapan:")
        
        explanations = {
            "🔤 Teks Asli": "Teks input yang belum diproses",
            "🧹 Pembersihan": "Menghapus URL, mention, hashtag, angka, emoji, dan karakter khusus",
            "📝 Case Folding": "Mengubah semua huruf menjadi huruf kecil untuk konsistensi",
            "✂️ Tokenisasi": "Memecah teks menjadi token/kata individual",
            "🚫 Hapus Stopwords": "Menghapus kata-kata umum bahasa Indonesia yang tidak bermakna",
            "🔄 Normalisasi": "Mengubah singkatan dan slang menjadi bentuk baku (contoh: 'bgt' → 'sangat')",
            "🌱 Stemming": "Mengubah kata ke bentuk dasarnya menggunakan algoritma Sastrawi",
            "✅ Hasil Akhir": "Teks yang sudah siap untuk dianalisis oleh model machine learning"
        }
        
        for stage, explanation in explanations.items():
            st.write(f"**{stage}**: {explanation}")
        
        # Tips untuk preprocessing
        st.subheader("💡 Tips Preprocessing Bahasa Indonesia:")
        st.info("""
        - **Normalisasi** sangat penting untuk bahasa Indonesia karena banyaknya singkatan dan slang
        - **Stemming** menggunakan algoritma Sastrawi yang dirancang khusus untuk bahasa Indonesia
        - **Stopwords** disesuaikan dengan kata-kata umum bahasa Indonesia
        - **Cleaning** menghapus noise seperti emoji dan karakter khusus yang sering muncul di media sosial
        """)

if __name__ == "__main__":
    main()
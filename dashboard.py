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

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
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
    sentiment_counts = df['sentimen'].value_counts()
    labels = ['Negatif', 'Positif']
    values = [sentiment_counts[0], sentiment_counts[1]]
    colors = ['#ff7f7f', '#7fbf7f']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.3,
        marker=dict(colors=colors)
    )])
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=12
    )
    
    fig.update_layout(
        title="Distribusi Sentimen",
        font=dict(size=14),
        showlegend=True
    )
    
    return fig

def create_wordcloud(text, title, colormap='viridis'):
    """Membuat word cloud"""
    if len(text) == 0:
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100
    ).generate(' '.join(text))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def get_top_words(texts, n=20):
    """Mendapatkan kata-kata yang paling sering muncul"""
    all_words = []
    for text in texts:
        if pd.notna(text):
            all_words.extend(text.split())
    
    word_freq = Counter(all_words)
    return word_freq.most_common(n)

def show_preprocessing_steps(analyzer, text):
    """Menampilkan tabel langkah-langkah preprocessing"""
    if text.strip():
        processed_text, steps = analyzer.preprocess_text(text, show_steps=True)
        
        st.subheader("Tahapan Preprocessing")
        
        # Buat DataFrame untuk menampilkan steps
        steps_df = pd.DataFrame([
            ["Original", steps['original']],
            ["Cleaned", steps['cleaned']],
            ["Case Folded", steps['casefolded']],
            ["Tokenized", str(steps['tokenized'])],
            ["No Stopwords", str(steps['no_stopwords'])],
            ["Normalized", str(steps['normalized'])],
            ["Stemmed", str(steps['stemmed'])],
            ["Final", steps['final']]
        ], columns=["Tahap", "Hasil"])
        
        st.dataframe(steps_df, use_container_width=True)
        
        return processed_text
    return None

def main():
    """Fungsi utama dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data dan model
    df = load_data()
    analyzer = load_model()
    
    if df is None or analyzer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["📈 Dashboard Utama", "🔮 Prediksi Sentimen", "📋 Preprocessing Demo"]
    )
    
    if page == "📈 Dashboard Utama":
        show_main_dashboard(df)
    elif page == "🔮 Prediksi Sentimen":
        show_prediction_page(analyzer)
    elif page == "📋 Preprocessing Demo":
        show_preprocessing_demo(analyzer)

def show_main_dashboard(df):
    """Menampilkan dashboard utama"""
    st.header("Dashboard Analisis Sentimen")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_data = len(df)
    positive_count = len(df[df['sentimen'] == 1])
    negative_count = len(df[df['sentimen'] == 0])
    positive_ratio = (positive_count / total_data) * 100
    
    with col1:
        st.metric("Total Data", total_data)
    with col2:
        st.metric("Sentimen Positif", positive_count)
    with col3:
        st.metric("Sentimen Negatif", negative_count)
    with col4:
        st.metric("Rasio Positif", f"{positive_ratio:.1f}%")
    
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
            fig_bar = go.Figure([go.Bar(x=list(counts), y=list(words), orientation='h')])
            fig_bar.update_layout(
                title="15 Kata Teratas",
                xaxis_title="Frekuensi",
                yaxis_title="Kata",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Word Clouds
    st.header("Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentimen Positif")
        positive_texts = df[df['sentimen'] == 1]['processed_text'].dropna().tolist()
        if positive_texts:
            fig_wc_pos = create_wordcloud(positive_texts, "Word Cloud Sentimen Positif", 'Greens')
            if fig_wc_pos:
                st.pyplot(fig_wc_pos, clear_figure=True)
        else:
            st.write("Tidak ada data sentimen positif")
    
    with col2:
        st.subheader("Sentimen Negatif")
        negative_texts = df[df['sentimen'] == 0]['processed_text'].dropna().tolist()
        if negative_texts:
            fig_wc_neg = create_wordcloud(negative_texts, "Word Cloud Sentimen Negatif", 'Reds')
            if fig_wc_neg:
                st.pyplot(fig_wc_neg, clear_figure=True)
        else:
            st.write("Tidak ada data sentimen negatif")
    
    # Top words by sentiment
    st.header("Kata-kata Teratas per Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Kata Sentimen Positif")
        if positive_texts:
            top_words_pos = get_top_words(positive_texts, 15)
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
        st.subheader("Top 15 Kata Sentimen Negatif")
        if negative_texts:
            top_words_neg = get_top_words(negative_texts, 15)
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
    st.header("Sample Data")
    st.dataframe(df[['text', 'sentimen', 'processed_text']].head(10), use_container_width=True)

def show_prediction_page(analyzer):
    """Halaman prediksi sentimen"""
    st.header("🔮 Prediksi Sentimen")
    
    st.write("Masukkan teks untuk memprediksi sentimennya:")
    
    # Input teks
    user_input = st.text_area(
        "Teks untuk dianalisis:",
        height=100,
        placeholder="Contoh: Saya sangat senang dengan pelayanan yang diberikan..."
    )
    
    if st.button("Analisis Sentimen", type="primary"):
        if user_input.strip():
            # Prediksi
            result = analyzer.predict_sentiment(user_input)
            
            # Tampilkan hasil
            col1, col2 = st.columns(2)
            
            with col1:
                # Hasil prediksi
                sentiment_color = "green" if result['sentiment'] == "Positif" else "red"
                st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid {sentiment_color}; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: {sentiment_color}; margin: 0;">Sentimen: {result['sentiment']}</h3>
                    <p style="margin: 0;">Confidence: {result['confidence']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Probability chart
                fig = go.Figure([go.Bar(
                    x=['Negatif', 'Positif'],
                    y=[result['probability_negative'], result['probability_positive']],
                    marker_color=['lightcoral', 'lightgreen']
                )])
                fig.update_layout(
                    title="Probabilitas Sentimen",
                    yaxis_title="Probabilitas",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Preprocessing steps
            show_preprocessing_steps(analyzer, user_input)
            
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

def show_preprocessing_demo(analyzer):
    """Halaman demo preprocessing"""
    st.header("📋 Demo Preprocessing")
    
    st.write("Lihat bagaimana teks diproses melalui setiap tahap preprocessing:")
    
    # Input teks
    demo_text = st.text_area(
        "Masukkan teks untuk melihat proses preprocessing:",
        height=100,
        placeholder="Contoh: I really LOVE this product!!! It's amazing 😊 http://example.com",
        value="I really LOVE this product!!! It's amazing 😊 http://example.com"
    )
    
    if demo_text.strip():
        show_preprocessing_steps(analyzer, demo_text)
        
        # Penjelasan setiap tahap
        st.subheader("Penjelasan Tahapan:")
        
        explanations = {
            "Text Cleaning": "Menghapus URL, mention, hashtag, angka, dan karakter khusus",
            "Case Folding": "Mengubah semua huruf menjadi huruf kecil",
            "Tokenizing": "Memecah teks menjadi token/kata individual",
            "Stopwords Removal": "Menghapus kata-kata umum yang tidak bermakna",
            "Normalization": "Mengubah singkatan menjadi bentuk lengkap",
            "Stemming": "Mengubah kata ke bentuk dasarnya",
            "Final": "Hasil akhir yang siap untuk dianalisis"
        }
        
        for stage, explanation in explanations.items():
            st.write(f"**{stage}**: {explanation}")

if __name__ == "__main__":
    main()
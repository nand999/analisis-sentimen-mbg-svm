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
from dateutil import parser as date_parser
from datetime import datetime
import re
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

def detect_date_column(df):
    """Mendeteksi kolom tanggal dari berbagai nama kolom yang umum"""
    # Daftar nama kolom yang umum untuk tanggal
    date_column_names = [
        'created_at', 'date', 'timestamp', 'time', 'datetime', 'created', 
        'posted_at', 'published_at', 'tanggal', 'waktu', 'created_date',
        'post_date', 'tweet_created_at', 'creation_date', 'date_created'
    ]
    
    # Cek kolom yang ada di dataframe (case-insensitive)
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in date_column_names:
            return col
    
    # Jika tidak ditemukan dari nama, coba deteksi dari isi kolom
    for col in df.columns:
        if df[col].dtype == 'object':  # Hanya cek kolom string
            # Ambil sample non-null values
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                # Coba parse beberapa nilai
                parsed_count = 0
                for val in sample_values:
                    if parse_date_robust(str(val)) is not None:
                        parsed_count += 1
                
                # Jika lebih dari 70% bisa diparsing sebagai tanggal, anggap sebagai kolom tanggal
                if parsed_count / len(sample_values) > 0.7:
                    return col
    
    return None

def parse_date_robust(date_string):
    """Parse tanggal dengan berbagai format secara robust"""
    if pd.isna(date_string) or not date_string or str(date_string).strip() == '':
        return None
    
    date_string = str(date_string).strip()
    
    try:
        # Coba parse dengan dateutil parser (sangat fleksibel)
        parsed_date = date_parser.parse(date_string, fuzzy=True)
        return parsed_date
    except:
        pass
    
    # Coba format-format khusus yang mungkin tidak ditangani dateutil
    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%d %b %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%B %d, %Y',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f',
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except:
            continue
    
    return None

def get_date_range(df, date_column):
    """Mendapatkan rentang tanggal dari kolom tanggal"""
    if date_column is None or date_column not in df.columns:
        return None, None
    
    # Parse semua tanggal
    dates = []
    for val in df[date_column].dropna():
        parsed = parse_date_robust(val)
        if parsed:
            dates.append(parsed)
    
    if not dates:
        return None, None
    
    min_date = min(dates)
    max_date = max(dates)
    
    return min_date, max_date

def get_temporal_statistics(df, date_column):
    """Mendapatkan statistik temporal (per tahun dan per bulan)"""
    if date_column is None or date_column not in df.columns:
        return None
    
    # Parse semua tanggal
    dates = []
    for val in df[date_column].dropna():
        parsed = parse_date_robust(val)
        if parsed:
            dates.append(parsed)
    
    if not dates:
        return None
    
    # Buat DataFrame dari dates untuk analisis
    df_dates = pd.DataFrame({'date': dates})
    
    # Ekstrak tahun dan bulan
    df_dates['year'] = df_dates['date'].dt.year
    df_dates['month'] = df_dates['date'].dt.month
    df_dates['year_month'] = df_dates['date'].dt.to_period('M').astype(str)
    
    # Hitung per tahun
    yearly_counts = df_dates['year'].value_counts().sort_index()
    
    # Hitung per bulan
    monthly_counts = df_dates['year_month'].value_counts().sort_index()
    
    # Konversi bulan ke format yang lebih readable
    month_names_id = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
        7: 'Jul', 8: 'Agt', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
    }
    
    monthly_labels = []
    for ym in monthly_counts.index:
        year, month = ym.split('-')
        month_name = month_names_id[int(month)]
        monthly_labels.append(f"{month_name} {year}")
    
    return {
        'yearly_counts': yearly_counts,
        'monthly_counts': monthly_counts,
        'monthly_labels': monthly_labels,
        'total_dates': len(dates)
    }

def create_temporal_charts(temporal_stats):
    """Membuat visualisasi distribusi temporal"""
    if temporal_stats is None:
        return None, None
    
    # Chart distribusi per tahun - Filter hanya tahun yang valid
    yearly_counts = temporal_stats['yearly_counts']
    
    # Filter: hanya ambil tahun yang berupa integer dan dalam range 1900-2100
    valid_years = []
    valid_counts = []
    for year, count in yearly_counts.items():
        try:
            year_int = int(year)
            if 1900 <= year_int <= 2100:
                valid_years.append(str(year_int))
                valid_counts.append(count)
        except (ValueError, TypeError):
            # Skip tahun yang tidak valid (bukan integer atau di luar range)
            continue
    
    fig_yearly = go.Figure([go.Bar(
        x=valid_years,
        y=valid_counts,
        marker_color='lightblue',
        text=valid_counts,
        textposition='auto',
        hovertemplate='<b>Tahun %{x}</b><br>Jumlah: %{y}<extra></extra>'
    )])
    
    fig_yearly.update_layout(
        title={
            'text': "Distribusi Data per Tahun",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Tahun",
        yaxis_title="Jumlah Data",
        xaxis={'type': 'category'},  # Set as category to prevent decimal years
        height=400,
        showlegend=False,
        hovermode='x'
    )
    
    # Chart distribusi per bulan
    monthly_counts = temporal_stats['monthly_counts']
    monthly_labels = temporal_stats['monthly_labels']
    
    fig_monthly = go.Figure([go.Bar(
        x=monthly_labels,
        y=monthly_counts.values,
        marker_color='lightcoral',
        text=monthly_counts.values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Jumlah: %{y}<extra></extra>'
    )])
    
    fig_monthly.update_layout(
        title={
            'text': "Distribusi Data per Bulan",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Bulan",
        yaxis_title="Jumlah Data",
        height=400,
        showlegend=False,
        hovermode='x',
        xaxis={'tickangle': -45}
    )
    
    return fig_yearly, fig_monthly

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
            ["🔄 Normalisasi", str(steps['normalized'])],
            ["🚫 Hapus Stopwords", str(steps['no_stopwords'])],
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
        
        # Render tabel dengan HTML agar bisa discroll horizontal
        html_table = styled_df.to_html(escape=False)
        st.markdown(
            f"""
            <div style="overflow-x: auto; max-width: 100%;">
                {html_table}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return processed_text
    return None



def show_examples():
    """Menampilkan contoh-contoh teks untuk analisis"""
    st.subheader("💡 Contoh Teks untuk Dicoba")
    
    examples = [
        "MBG program yang bermanfaat untuk generasi emas!",
        "Makan Bergizi Gratis merupakan bentuk komiitmen pemerintah dalam perbaikan gizi",
        "makan bergizi gratis ini bagus banget",
        "MBG program buang buang anggaran",
        "Makan Bergizi Gratis menjadi Makan Beracun Gratis",
        "MBG menjadi ladang korupsi",
        "Stop mbg sebelum jatuh lebih banyak korban keracunan",
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Contoh {i}: {example[:50]}...", key=f"example_{i}"):
            return example
    
    return None

def main():
    """Fungsi utama dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Analisis Sentimen Program MBG Bahasa Indonesia</h1>', unsafe_allow_html=True)
    
    # Load data dan model
    df = load_data()
    analyzer = load_model()
    
    if df is None or analyzer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🧭 Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["📈 Dashboard Utama", "🔮 Prediksi Sentimen", "📋 Demo Preprocessing", "📁 Analisis CSV"]
    )
    
    if page == "📈 Dashboard Utama":
        show_main_dashboard(df)
    elif page == "🔮 Prediksi Sentimen":
        show_prediction_page(analyzer)
    elif page == "📋 Demo Preprocessing":
        show_preprocessing_demo(analyzer)
    elif page == "📁 Analisis CSV":
        show_csv_analysis_page(analyzer)

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
    
    # Dataset Display Section
    st.markdown("---")
    st.header("📋 Dataset yang Digunakan untuk Training Model")
    
    try:
        # Load dataset kolom_lengkap.csv
        dataset_path = 'data_mbg_labelled.csv'
        df_dataset = pd.read_csv(dataset_path)
        
        # Pilih kolom yang dibutuhkan
        if all(col in df_dataset.columns for col in ['created_at', 'text', 'sentiment']):
            display_df = df_dataset[['created_at', 'text', 'sentiment']].copy()
            
            # Map sentiment ke label yang lebih readable
            display_df['sentiment'] = display_df['sentiment'].map({0: 'Negatif', 1: 'Positif'})
            
            # Rename columns untuk display yang lebih baik
            display_df = display_df.rename(columns={
                'created_at': 'Tanggal Dibuat',
                'text': 'Teks',
                'sentiment': 'Sentimen'
            })
            
            # Tampilkan tabel tanpa width constraint agar text bisa wrap
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.warning("⚠️ Kolom yang diperlukan (created_at, text, sentiment) tidak ditemukan dalam dataset.")
    except FileNotFoundError:
        st.error(f"❌ File dataset '{dataset_path}' tidak ditemukan.")
    except Exception as e:
        st.error(f"❌ Error saat memuat dataset: {str(e)}")
    
    # Model Performance Section
    st.markdown("---")
    st.header("📊 Performa Model")
    
    try:
        import json
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("📈 Precision (Weighted)", f"{metrics['classification_report']['weighted_avg']['precision']:.2%}")
        with col3:
            st.metric("📊 Recall (Weighted)", f"{metrics['classification_report']['weighted_avg']['recall']:.2%}")
        with col4:
            st.metric("⚖️ F1-Score (Weighted)", f"{metrics['classification_report']['weighted_avg']['f1-score']:.2%}")
        
        st.info(f"ℹ️ Model dilatih dengan rasio train:test = **{metrics['best_ratio']}**")
        
        # Confusion Matrix and Classification Report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Confusion Matrix")
            
            cm = metrics['confusion_matrix']
            
            # Create confusion matrix visualization using plotly
            import plotly.figure_factory as ff
            
            z = [[cm['true_negative'], cm['false_positive']], 
                 [cm['false_negative'], cm['true_positive']]]
            
            x = ['Prediksi Negatif', 'Prediksi Positif']
            y = ['Aktual Negatif', 'Aktual Positif']
            
            z_text = [[f"TN: {cm['true_negative']}", f"FP: {cm['false_positive']}"],
                      [f"FN: {cm['false_negative']}", f"TP: {cm['true_positive']}"]]
            
            fig = ff.create_annotated_heatmap(
                z, 
                x=x, 
                y=y, 
                annotation_text=z_text,
                colorscale='Blues',
                showscale=True
            )
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Prediksi",
                yaxis_title="Aktual",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown("""
            **Penjelasan:**
            - **TN (True Negative)**: Prediksi Negatif, Aktual Negatif ✅
            - **TP (True Positive)**: Prediksi Positif, Aktual Positif ✅
            - **FP (False Positive)**: Prediksi Positif, Aktual Negatif ❌ (Error Tipe I)
            - **FN (False Negative)**: Prediksi Negatif, Aktual Positif ❌ (Error Tipe II)
            """)
        
        with col2:
            st.subheader("📋 Classification Report")
            
            # Create dataframe for classification report
            report_data = []
            
            for label in ['negative', 'positive']:
                label_name = 'Negatif' if label == 'negative' else 'Positif'
                report_data.append({
                    'Kelas': label_name,
                    'Precision': f"{metrics['classification_report'][label]['precision']:.4f}",
                    'Recall': f"{metrics['classification_report'][label]['recall']:.4f}",
                    'F1-Score': f"{metrics['classification_report'][label]['f1-score']:.4f}"
                })
            
            # Add weighted average
            report_data.append({
                'Kelas': 'Weighted Avg',
                'Precision': f"{metrics['classification_report']['weighted_avg']['precision']:.4f}",
                'Recall': f"{metrics['classification_report']['weighted_avg']['recall']:.4f}",
                'F1-Score': f"{metrics['classification_report']['weighted_avg']['f1-score']:.4f}"
            })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            # Metrics explanation
            st.markdown("""
            **Penjelasan Metrik:**
            - **Precision**: Dari semua prediksi positif/negatif, berapa persen yang benar?
            - **Recall**: Dari semua data aktual positif/negatif, berapa persen yang berhasil diprediksi?
            - **F1-Score**: Harmonic mean dari Precision dan Recall (keseimbangan keduanya)
            - **Weighted Avg**: Rata-rata tertimbang berdasarkan jumlah sampel per kelas
            """)
            
            # Bar chart for metrics comparison
            fig_metrics = go.Figure()
            
            # Map display names to JSON keys
            metric_mapping = {
                'Precision': 'precision',
                'Recall': 'recall',
                'F1-Score': 'f1-score'
            }
            
            for metric_display, metric_key in metric_mapping.items():
                values = [
                    metrics['classification_report']['negative'][metric_key],
                    metrics['classification_report']['positive'][metric_key]
                ]
                fig_metrics.add_trace(go.Bar(
                    name=metric_display,
                    x=['Negatif', 'Positif'],
                    y=values,
                    text=[f"{v:.2%}" for v in values],
                    textposition='auto'
                ))
            
            fig_metrics.update_layout(
                title="Perbandingan Metrik per Kelas",
                xaxis_title="Kelas",
                yaxis_title="Score",
                barmode='group',
                height=350,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("⚠️ File model_metrics.json tidak ditemukan. Jalankan sentiment_model.py terlebih dahulu untuk menghasilkan metrik performa model.")
    except Exception as e:
        st.error(f"❌ Error saat memuat metrik model: {str(e)}")

def show_prediction_page(analyzer):
    """Halaman prediksi sentimen"""
    st.header("🔮 Prediksi Sentimen")
    
    st.write("Masukkan teks bahasa Indonesia untuk memprediksi sentimennya:")
    
    # # Contoh teks
    # selected_example = show_examples()
    
    # Input teks
    default_text = "MBG bermanfaat bagi anak-anak dan orangtua"
    user_input = st.text_area(
        "Teks untuk dianalisis:",
        height=100,
        value=default_text,
        placeholder="Contoh: Pelayanannya sangat memuaskan dan staffnya ramah sekali!"
    )
    
    if st.button("🚀 Analisis Sentimen", type="primary"):
        # Validasi input kosong
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            # Cek apakah input hanya simbol/tanda baca
            cleaned_for_check = re.sub(r'[^\w\s]', '', user_input).strip()
            if not cleaned_for_check:
                st.error("⚠️ Input hanya berisi simbol atau tanda baca. Tidak dapat melakukan analisis sentimen karena tidak ada teks bermakna.")
            else:
                with st.spinner("Menganalisis sentimen..."):
                    # Prediksi
                    result = analyzer.predict_sentiment(user_input)
                    
                    # Cek jika hasil tidak dapat ditentukan
                    if result['sentiment'] == 'Tidak dapat menentukan':
                        st.warning("⚠️ Tidak dapat menentukan sentimen. Teks mungkin terlalu pendek atau tidak mengandung kata bermakna setelah preprocessing.")
                    
                    # Tampilkan hasil
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Hasil prediksi
                        if result['sentiment'] == 'Tidak dapat menentukan':
                            sentiment_color = "gray"
                            sentiment_icon = "😐"
                        else:
                            sentiment_color = "green" if result['sentiment'] == "Positif" else "red"
                            sentiment_icon = "😊" if result['sentiment'] == "Positif" else "😞"
                        
                        st.markdown(f"""
                        <div style="padding: 2rem; border: 3px solid {sentiment_color}; border-radius: 1rem; text-align: center; background-color: rgba({'128,128,128' if sentiment_color == 'gray' else '0,255,0' if sentiment_color == 'green' else '255,0,0'}, 0.1);">
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


def show_preprocessing_demo(analyzer):
    """Halaman demo preprocessing"""
    st.header("📋 Demo Preprocessing")
    
    st.write("Lihat bagaimana teks bahasa Indonesia diproses melalui setiap tahap preprocessing:")
    
    # Input teks
    demo_text = st.text_area(
        "Masukkan teks untuk melihat proses preprocessing:",
        height=100,
        placeholder="Contoh: MBG sangat bermanfaat bagi anak-anak dan orangtua!😊",
        value="MBG sangat bermanfaat bagi anak-anak dan orangtua!😊"
    )
    
    # Hanya tampilkan jika ada input yang tidak kosong
    if demo_text.strip():
        # Cek apakah input hanya simbol/tanda baca
        cleaned_for_check = re.sub(r'[^\w\s]', '', demo_text).strip()
        if not cleaned_for_check:
            st.warning("⚠️ Input hanya berisi simbol atau tanda baca. Sistem akan tetap memproses tetapi hasil mungkin kosong setelah pembersihan.")
        
        show_preprocessing_steps(analyzer, demo_text)
        
        # Penjelasan setiap tahap
        st.subheader("📚 Penjelasan Tahapan:")
        
        explanations = {
            "🔤 Teks Asli": "Teks input yang belum diproses",
            "🧹 Pembersihan": "Menghapus URL, mention, hashtag, angka, emoji, dan karakter khusus",
            "📝 Case Folding": "Mengubah semua huruf menjadi huruf kecil untuk konsistensi",
            "✂️ Tokenisasi": "Memecah teks menjadi token/kata individual",
            "🔄 Normalisasi": "Mengubah singkatan dan slang menjadi bentuk baku (contoh: 'bgt' → 'sangat')",
            "🚫 Hapus Stopwords": "Menghapus kata-kata umum bahasa Indonesia yang tidak bermakna",
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
    else:
        st.info("ℹ️ Masukkan teks untuk melihat tahapan preprocessing.")


def process_csv_predictions(analyzer, df_csv, text_column, confidence_threshold=0.5):
    """Memproses prediksi sentimen untuk CSV dengan filtering confidence"""
    results = []
    
    for idx, row in df_csv.iterrows():
        text = str(row[text_column])
        if pd.isna(text) or not text.strip():
            continue
            
        # Prediksi sentimen
        prediction = analyzer.predict_sentiment(text)
        
        # Simpan hasil
        results.append({
            'text': text,
            'sentiment': prediction['sentiment'],
            'confidence': prediction['confidence'],
            'probability_negative': prediction['probability_negative'],
            'probability_positive': prediction['probability_positive']
        })
    
    # Buat DataFrame hasil
    df_results = pd.DataFrame(results)
    
    # Filter berdasarkan confidence threshold
    df_confident = df_results[df_results['confidence'] >= confidence_threshold].copy()
    df_neutral = df_results[df_results['confidence'] < confidence_threshold].copy()
    
    return df_confident, df_neutral

def create_csv_pie_chart(df_results):
    """Membuat pie chart distribusi sentimen untuk hasil CSV"""
    if df_results.empty:
        return None
        
    sentiment_counts = df_results['sentiment'].value_counts()
    labels = []
    values = []
    colors = []
    
    if 'Negatif' in sentiment_counts.index:
        labels.append('Negatif')
        values.append(sentiment_counts['Negatif'])
        colors.append('#ff7f7f')
    
    if 'Positif' in sentiment_counts.index:
        labels.append('Positif')
        values.append(sentiment_counts['Positif'])
        colors.append('#7fbf7f')
    
    if not labels:
        return None
    
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
            'text': "Distribusi Sentimen CSV",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig

def show_csv_analysis_page(analyzer):
    """Halaman analisis CSV"""
    st.header("📁 Analisis Sentimen dari CSV")
    
    st.write("""
    Upload file CSV untuk melakukan prediksi sentimen secara batch. 
    File CSV harus memiliki kolom **'text'** atau **'full_text'** yang berisi teks untuk dianalisis.
    """)
    
    # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=['csv'],
        help="File CSV harus memiliki kolom 'text' atau 'full_text'"
    )
    
    if uploaded_file is not None:
        try:
            # Baca CSV
            df_csv = pd.read_csv(uploaded_file)
            
            # Validasi CSV tidak kosong
            if len(df_csv) == 0:
                st.warning("⚠️ File CSV tidak memiliki data. Silakan upload file dengan data yang valid.")
                return
            
            st.success(f"✅ File berhasil diupload! Total baris: {len(df_csv):,}")
            
            # Deteksi kolom tanggal dan analisis temporal
            date_column = detect_date_column(df_csv)
            if date_column:
                min_date, max_date = get_date_range(df_csv, date_column)
                temporal_stats = get_temporal_statistics(df_csv, date_column)
                
                if min_date and max_date and temporal_stats:
                    # Tampilkan informasi rentang waktu
                    st.markdown("---")
                    st.subheader("📅 Informasi Temporal Data")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info(f"**Rentang Waktu:** {min_date.strftime('%d %B %Y, %H:%M:%S')} sampai {max_date.strftime('%d %B %Y, %H:%M:%S')}")
                    with col2:
                        st.metric("📊 Total Data Bertanggal", f"{temporal_stats['total_dates']:,}")
                    
                    st.caption(f"Kolom tanggal yang terdeteksi: **{date_column}**")
                    
                    # Tampilkan chart temporal
                    fig_yearly, fig_monthly = create_temporal_charts(temporal_stats)
                    
                    if fig_yearly and fig_monthly:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    st.markdown("---")
                else:
                    st.warning(f"⚠️ Kolom tanggal terdeteksi ({date_column}) tetapi tidak dapat mem-parsing tanggal")
            else:
                st.info("ℹ️ Tidak ada kolom tanggal yang terdeteksi dalam CSV")
            
            # Validasi kolom
            text_column = None
            if 'text' in df_csv.columns:
                text_column = 'text'
            elif 'full_text' in df_csv.columns:
                text_column = 'full_text'
            else:
                st.error("❌ File CSV harus memiliki kolom 'text' atau 'full_text'")
                st.info(f"Kolom yang tersedia: {', '.join(df_csv.columns)}")
                return
            
            st.info(f"📝 Menggunakan kolom teks: **{text_column}**")
            
            # Tampilkan preview
            with st.expander("👀 Preview Data CSV"):
                st.dataframe(df_csv.head(10), use_container_width=True)
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold ",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Prediksi dengan confidence di bawah threshold ini tidak dimasukkan dalam analisis"
            )
            
            # Tombol proses
            if st.button("🚀 Proses Prediksi Sentimen", type="primary"):
                with st.spinner("Memproses prediksi sentimen..."):
                    # Proses prediksi
                    df_confident, df_neutral = process_csv_predictions(
                        analyzer, df_csv, text_column, confidence_threshold
                    )
                    
                    # Simpan ke session state
                    st.session_state['csv_results'] = df_confident
                    st.session_state['csv_neutral'] = df_neutral
                    st.session_state['csv_processed'] = True
                    
                st.success("✅ Prediksi selesai!")
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error membaca file CSV: {str(e)}")
            return
    
    # Tampilkan hasil jika sudah diproses
    if st.session_state.get('csv_processed', False):
        df_confident = st.session_state.get('csv_results')
        df_neutral = st.session_state.get('csv_neutral')
        
        if df_confident is not None:
            st.markdown("---")
            st.header("📊 Hasil Analisis")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_data = len(df_confident) + len(df_neutral)
            positive_count = len(df_confident[df_confident['sentiment'] == 'Positif'])
            negative_count = len(df_confident[df_confident['sentiment'] == 'Negatif'])
            neutral_count = len(df_neutral)
            
            with col1:
                st.metric("📄 Total Data", f"{total_data:,}")
            with col2:
                st.metric("😊 Sentimen Positif", f"{positive_count:,}")
            with col3:
                st.metric("😞 Sentimen Negatif", f"{negative_count:,}")
            with col4:
                st.metric("😐 Confidence Level < Threshold", f"{neutral_count:,}")
            
            # Visualisasi
            if not df_confident.empty:
                st.markdown("---")
                st.subheader("📈 Visualisasi Sentimen")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie Chart
                    fig_pie = create_csv_pie_chart(df_confident)
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar Chart distribusi confidence
                    fig_conf = go.Figure([go.Histogram(
                        x=df_confident['confidence'],
                        nbinsx=20,
                        marker_color='lightblue'
                    )])
                    fig_conf.update_layout(
                        title="Distribusi Confidence Score",
                        xaxis_title="Confidence",
                        yaxis_title="Frekuensi",
                        height=400
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Word Clouds
                st.markdown("---")
                st.subheader("☁️ Word Clouds")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**😊 Sentimen Positif**")
                    positive_texts = df_confident[df_confident['sentiment'] == 'Positif']['text'].tolist()
                    if positive_texts:
                        # Preprocess texts untuk wordcloud
                        processed_positive = [analyzer.preprocess_text(text) for text in positive_texts]
                        fig_wc_pos = create_wordcloud(processed_positive, "Word Cloud Sentimen Positif", 'Greens')
                        if fig_wc_pos:
                            st.pyplot(fig_wc_pos, clear_figure=True)
                    else:
                        st.info("Tidak ada data sentimen positif")
                
                with col2:
                    st.markdown("**😞 Sentimen Negatif**")
                    negative_texts = df_confident[df_confident['sentiment'] == 'Negatif']['text'].tolist()
                    if negative_texts:
                        # Preprocess texts untuk wordcloud
                        processed_negative = [analyzer.preprocess_text(text) for text in negative_texts]
                        fig_wc_neg = create_wordcloud(processed_negative, "Word Cloud Sentimen Negatif", 'Reds')
                        if fig_wc_neg:
                            st.pyplot(fig_wc_neg, clear_figure=True)
                    else:
                        st.info("Tidak ada data sentimen negatif")
                
                # Top words by sentiment
                st.markdown("---")
                st.subheader("📈 Kata-kata Teratas per Sentimen")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**😊 Top 15 Kata Sentimen Positif**")
                    if positive_texts:
                        processed_positive = [analyzer.preprocess_text(text) for text in positive_texts]
                        top_words_pos = get_top_words(processed_positive, 15)
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
                    st.markdown("**😞 Top 15 Kata Sentimen Negatif**")
                    if negative_texts:
                        processed_negative = [analyzer.preprocess_text(text) for text in negative_texts]
                        top_words_neg = get_top_words(processed_negative, 15)
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
            
            # Tabel hasil
            st.markdown("---")
            st.subheader("📋 Tabel Hasil Prediksi")
            
            if not df_confident.empty:
                # Format tabel untuk ditampilkan
                st.subheader("📊 Hasil Prediksi (Confident)")
                st.write(f"Total data dengan confidence ≥ {confidence_threshold}: **{len(df_confident):,}**")
                
                # Prepare display dataframe
                df_display = df_confident[['text', 'sentiment', 'confidence']].copy()
                df_display.columns = ['Teks', 'Sentimen', 'Confidence']
                
                # Tampilkan tabel tanpa width constraint agar text bisa wrap
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download hasil
                csv_download = df_confident.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_download,
                    file_name="hasil_prediksi_sentimen.csv",
                    mime="text/csv"
                )
                
                # ===== BAGIAN KESIMPULAN/SUMMARY =====
                st.markdown("---")
                st.subheader("📝 Kesimpulan Analisis Sentimen")
                
                # Hitung statistik untuk kesimpulan
                total_analyzed = len(df_confident) + len(df_neutral)
                pct_positive = (positive_count / total_analyzed * 100) if total_analyzed > 0 else 0
                pct_negative = (negative_count / total_analyzed * 100) if total_analyzed > 0 else 0
                pct_neutral = (neutral_count / total_analyzed * 100) if total_analyzed > 0 else 0
                
                avg_confidence = df_confident['confidence'].mean() if not df_confident.empty else 0
                
                # Tentukan sentimen dominan
                if positive_count > negative_count:
                    dominant_sentiment = "Positif"
                    dominant_icon = "😊"
                    dominant_color = "green"
                elif negative_count > positive_count:
                    dominant_sentiment = "Negatif"
                    dominant_icon = "😞"
                    dominant_color = "red"
                else:
                    dominant_sentiment = "Seimbang"
                    dominant_icon = "😐"
                    dominant_color = "gray"
                
                # Build interpretasi list
                interpretasi_list = []
                
                # Interpretasi berdasarkan distribusi sentimen
                if pct_positive > 70:
                    interpretasi_list.append("Data menunjukkan **sentimen sangat positif** dengan lebih dari 70% respons positif.")
                    interpretasi_list.append("Ini mengindikasikan tingkat kepuasan atau penerimaan yang sangat baik.")
                elif pct_positive > 50:
                    interpretasi_list.append("Data menunjukkan **sentimen cenderung positif** dengan mayoritas respons positif.")
                    interpretasi_list.append("Secara umum, terdapat penerimaan yang baik meskipun masih ada ruang untuk perbaikan.")
                elif pct_negative > 70:
                    interpretasi_list.append("Data menunjukkan **sentimen sangat negatif** dengan lebih dari 70% respons negatif.")
                    interpretasi_list.append("Ini mengindikasikan adanya masalah serius yang perlu segera ditangani.")
                elif pct_negative > 50:
                    interpretasi_list.append("Data menunjukkan **sentimen cenderung negatif** dengan mayoritas respons negatif.")
                    interpretasi_list.append("Diperlukan perhatian khusus untuk meningkatkan kualitas atau layanan.")
                else:
                    interpretasi_list.append("Data menunjukkan **sentimen yang seimbang** antara positif dan negatif.")
                    interpretasi_list.append("Terdapat opini yang beragam, menunjukkan pengalaman yang bervariasi.")
                
                # Interpretasi berdasarkan confidence
                if avg_confidence > 0.8:
                    interpretasi_list.append(f"Confidence score yang tinggi ({avg_confidence:.1%}) menunjukkan model sangat yakin dengan prediksinya.")
                    interpretasi_list.append("Hasil analisis dapat diandalkan untuk pengambilan keputusan.")
                elif avg_confidence > 0.6:
                    interpretasi_list.append(f"Confidence score yang cukup baik ({avg_confidence:.1%}) menunjukkan prediksi yang dapat diandalkan.")
                    interpretasi_list.append("Sebagian besar hasil analisis dapat dipercaya.")
                else:
                    interpretasi_list.append(f"Confidence score yang moderat ({avg_confidence:.1%}) menunjukkan beberapa prediksi mungkin ambigu.")
                    interpretasi_list.append("Disarankan untuk melakukan validasi manual pada data dengan confidence rendah.")
                
                # Interpretasi data netral
                if pct_neutral > 30:
                    interpretasi_list.append(f"Terdapat **{pct_neutral:.1f}%** data dengan confidence level < threshold.")
                    interpretasi_list.append("Data dengan confidence level < threshold yang tinggi mungkin mengindikasikan teks yang ambigu atau memerlukan konteks tambahan.")
                
                # Build rekomendasi list
                rekomendasi_list = []
                
                if pct_negative > 40:
                    rekomendasi_list.append("Identifikasi tema atau topik utama dari sentimen negatif untuk perbaikan.")
                    rekomendasi_list.append("Lakukan analisis lebih lanjut pada kata-kata yang sering muncul di sentimen negatif.")
                
                if pct_neutral > 20:
                    rekomendasi_list.append("Review data dengan confidence level < threshold secara manual untuk memahami konteks yang lebih baik.")
                    rekomendasi_list.append("Pertimbangkan untuk meningkatkan threshold confidence jika diperlukan hasil yang lebih pasti.")
                
                rekomendasi_list.append("Gunakan word cloud dan top words untuk memahami topik yang paling sering dibahas.")
                rekomendasi_list.append("Pantau tren sentimen secara berkala untuk melihat perubahan dari waktu ke waktu.")
                
                # Tampilkan kesimpulan dalam card dengan st.container
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid {dominant_color};">
                        <h4 style="margin-top: 0; color: {dominant_color};">{dominant_icon} Sentimen Dominan: {dominant_sentiment}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Ringkasan Hasil Analisis:**")
                    st.markdown(f"- Dari **{total_analyzed:,}** data yang dianalisis, terdapat:")
                    st.markdown(f"  - **{positive_count:,}** sentimen positif ({pct_positive:.1f}%)")
                    st.markdown(f"  - **{negative_count:,}** sentimen negatif ({pct_negative:.1f}%)")
                    st.markdown(f"  - **{neutral_count:,}** data dengan confidence level < threshold ({pct_neutral:.1f}%)")
                    st.markdown(f"- Rata-rata confidence score: **{avg_confidence:.1%}**")
                    
                    st.markdown("**Interpretasi:**")
                    for item in interpretasi_list:
                        st.markdown(f"- {item}")
                    
                    st.markdown("**Rekomendasi:**")
                    for item in rekomendasi_list:
                        st.markdown(f"- {item}")
                
            else:
                st.warning("⚠️ Tidak ada data dengan confidence di atas threshold")
            
            # Tabel data netral
            if not df_neutral.empty:
                st.markdown("---")
                st.subheader(f"😐 Data (Confidence < threshold) - Total: {len(df_neutral):,}")
                
                with st.expander(f"📋 Data dengan Confidence Level < Threshold ({confidence_threshold})"):
                    st.write(f"Total data dengan confidence level < threshold: **{len(df_neutral):,}**")
                    st.caption("Data ini memiliki confidence di bawah threshold dan dianggap tidak pasti")
                    
                    # Prepare display dataframe
                    df_neutral_display = df_neutral[['text', 'sentiment', 'confidence']].copy()
                    df_neutral_display.columns = ['Teks', 'Sentimen', 'Confidence']
                    
                    # Tampilkan tabel tanpa width constraint agar text bisa wrap
                    st.dataframe(
                        df_neutral_display,
                        use_container_width=True,
                        height=300
                    )
                    
                    # Download data netral
                    csv_neutral_download = df_neutral.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data dengan Confidence Score dibawah Threshold (CSV)",
                        data=csv_neutral_download,
                        file_name="data_confidence_level_dibawah_threshold.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
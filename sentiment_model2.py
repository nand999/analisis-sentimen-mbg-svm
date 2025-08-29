import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK requirements
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        # Inisialisasi stemmer bahasa Indonesia
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Inisialisasi stopwords bahasa Indonesia
        stop_factory = StopWordRemoverFactory()
        self.stop_words = set(stop_factory.get_stop_words())
        
        # Tambahan stopwords khusus
        additional_stopwords = {
            'yg', 'dgn', 'nya', 'kalo', 'kalau', 'udah', 'udh', 'dah', 
            'lg', 'lagi', 'banget', 'bgt', 'emang', 'memang', 'sih',
            'aja', 'doang', 'nih', 'nah', 'lah', 'deh', 'dong', 'kok',
            'ya', 'yah', 'wkwk', 'haha', 'hihi', 'huhu', 'hehe'
        }
        self.stop_words.update(additional_stopwords)
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
        
        # SVM Model
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        
        # Kamus normalisasi bahasa Indonesia
        self.normalization_dict = {
            # Singkatan umum
            'yg': 'yang',
            'dgn': 'dengan',
            'krn': 'karena',
            'krna': 'karena',
            'tp': 'tapi',
            'tpi': 'tapi',
            'gk': 'tidak',
            'ga': 'tidak',
            'gak': 'tidak',
            'ngga': 'tidak',
            'nggak': 'tidak',
            'g': 'tidak',
            'tdk': 'tidak',
            'gitu': 'begitu',
            'gt': 'begitu',
            'gmn': 'bagaimana',
            'gimana': 'bagaimana',
            'dmn': 'dimana',
            'dimana': 'di mana',
            'kmn': 'kemana',
            'kemana': 'ke mana',
            'kapan': 'kapan',
            'knp': 'kenapa',
            'knapa': 'kenapa',
            'org': 'orang',
            'orng': 'orang',
            'tmn': 'teman',
            'temen': 'teman',
            'bgmn': 'bagaimana',
            'bgt': 'banget',
            'banget': 'sangat',
            'bener': 'benar',
            'bnr': 'benar',
            'bnyk': 'banyak',
            'bnyak': 'banyak',
            'udh': 'sudah',
            'udah': 'sudah',
            'dah': 'sudah',
            'telah': 'sudah',
            'blm': 'belum',
            'blom': 'belum',
            'msh': 'masih',
            'msih': 'masih',
            'lg': 'lagi',
            'lgi': 'lagi',
            'skrg': 'sekarang',
            'skrang': 'sekarang',
            'skg': 'sekarang',
            'nanti': 'nanti',
            'ntar': 'nanti',
            'tar': 'nanti',
            'bsk': 'besok',
            'besok': 'besok',
            'kmrn': 'kemarin',
            'kmarin': 'kemarin',
            'hrs': 'harus',
            'kudu': 'harus',
            'mesti': 'harus',
            'bs': 'bisa',
            'bsa': 'bisa',
            'isa': 'bisa',
            'biar': 'agar',
            'spy': 'agar',
            'supaya': 'agar',
            'kalo': 'kalau',
            'klo': 'kalau',
            'jd': 'jadi',
            'jadi': 'menjadi',
            'jdnya': 'jadinya',
            'jadinya': 'akhirnya',
            'jg': 'juga',
            'jga': 'juga',
            'jgn': 'jangan',
            'jangan': 'jangan',
            'jgn2': 'jangan-jangan',
            'aj': 'saja',
            'aja': 'saja',
            'doang': 'saja',
            'aje': 'saja',
            'cm': 'cuma',
            'cuma': 'hanya',
            'cman': 'hanya',
            'ckp': 'cukup',
            'cukup': 'cukup',
            'krg': 'kurang',
            'kurang': 'kurang',
            'emg': 'memang',
            'emang': 'memang',
            'mmg': 'memang',
            'sbnrnya': 'sebenarnya',
            'sbenernya': 'sebenarnya',
            'pdhl': 'padahal',
            'pdahal': 'padahal',
            'wlpn': 'walaupun',
            'walaupun': 'walaupun',
            'meskipun': 'walaupun',
            'walau': 'walaupun',
            'aplg': 'apalagi',
            'apalagi': 'apalagi',
            'mgkn': 'mungkin',
            'mungkin': 'mungkin',
            'mgkin': 'mungkin',
            'kyknya': 'kayaknya',
            'kyaknya': 'kayaknya',
            'kayaknya': 'sepertinya',
            'kyk': 'seperti',
            'kayak': 'seperti',
            'ky': 'seperti',
            'sprt': 'seperti',
            'kaya': 'seperti',
            'sy': 'saya',
            'gw': 'saya',
            'gue': 'saya',
            'w': 'saya',
            'aku': 'saya',
            'ak': 'saya',
            'km': 'kamu',
            'kmu': 'kamu',
            'lu': 'kamu',
            'lo': 'kamu',
            'elu': 'kamu',
            'elo': 'kamu',
            'u': 'kamu',
            'dy': 'dia',
            'dia': 'dia',
            'mrk': 'mereka',
            'mereka': 'mereka',
            'qt': 'kita',
            'kita': 'kita',
            'qta': 'kita',
            # Ekspresi perasaan
            'sedih': 'sedih',
            'seneng': 'senang',
            'senang': 'senang',
            'suka': 'suka',
            'sk': 'suka',
            'benci': 'benci',
            'kesel': 'kesal',
            'kesal': 'kesal',
            'marah': 'marah',
            'kecewa': 'kecewa',
            'puas': 'puas',
            'bahagia': 'bahagia',
            'kaget': 'kaget',
            'heran': 'heran',
            'bingung': 'bingung',
            'binun': 'bingung',
            'males': 'malas',
            'capek': 'capek',
            'cape': 'capek',
            'lelah': 'lelah',
            'tired': 'lelah',
            'stress': 'stres',
            'stres': 'stres',
            # Kata sifat umum
            'bagus': 'bagus',
            'jelek': 'jelek',
            'buruk': 'buruk',
            'parah': 'parah',
            'mantap': 'mantap',
            'mantul': 'mantap',
            'keren': 'keren',
            'gokil': 'keren',
            'ajib': 'keren',
            'top': 'bagus',
            'the best': 'terbaik',
            'terbaik': 'terbaik',
            'terburuk': 'terburuk',
            'worst': 'terburuk',
            'best': 'terbaik',
            'good': 'bagus',
            'bad': 'buruk',
            'nice': 'bagus',
            'awesome': 'keren',
            'amazing': 'menakjubkan',
            'terrible': 'buruk',
            'horrible': 'mengerikan',
            'excellent': 'sangat bagus',
            'perfect': 'sempurna',
            'ok': 'oke',
            'oke': 'baik',
            'okay': 'baik',
            'fine': 'baik',
            'normal': 'normal',
            'biasa': 'biasa',
            'lumayan': 'lumayan',
            'standard': 'standar',
            'standar': 'standar'
        }
        
    def text_cleaning(self, text):
        """Membersihkan teks dari karakter yang tidak diinginkan"""
        if pd.isna(text):
            return ""
        
        # Konversi ke string jika bukan string
        text = str(text)
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Hapus mention dan hashtag
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Hapus email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Hapus angka yang berdiri sendiri
        text = re.sub(r'\b\d+\b', '', text)
        
        # Hapus emoticon dan emoji
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Hapus karakter berulang (lebih dari 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def case_folding(self, text):
        """Mengubah teks menjadi huruf kecil"""
        return text.lower()
    
    def tokenizing(self, text):
        """Memecah teks menjadi token menggunakan split sederhana"""
        # Untuk bahasa Indonesia, kita bisa menggunakan split sederhana
        tokens = text.split()
        # Filter token yang terlalu pendek atau hanya karakter spesial
        tokens = [token for token in tokens if len(token) > 1 and token.isalpha()]
        return tokens
    
    def remove_stopwords(self, tokens):
        """Menghapus stopwords bahasa Indonesia"""
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def normalization(self, tokens):
        """Normalisasi kata bahasa Indonesia"""
        normalized_tokens = []
        for token in tokens:
            # Cek di kamus normalisasi
            if token in self.normalization_dict:
                normalized_tokens.append(self.normalization_dict[token])
            else:
                normalized_tokens.append(token)
        
        return normalized_tokens
    
    def stemming(self, tokens):
        """Melakukan stemming bahasa Indonesia menggunakan Sastrawi"""
        # Gabungkan tokens menjadi string untuk Sastrawi
        text = ' '.join(tokens)
        # Stem menggunakan Sastrawi
        stemmed_text = self.stemmer.stem(text)
        # Kembalikan ke bentuk tokens
        return stemmed_text.split()
    
    def preprocess_text(self, text, show_steps=False):
        """Melakukan preprocessing lengkap dengan opsi menampilkan setiap langkah"""
        steps = {}
        
        # Step 1: Text Cleaning
        cleaned = self.text_cleaning(text)
        if show_steps:
            steps['original'] = text
            steps['cleaned'] = cleaned
        
        # Step 2: Case Folding
        casefolded = self.case_folding(cleaned)
        if show_steps:
            steps['casefolded'] = casefolded
        
        # Step 3: Tokenizing
        tokens = self.tokenizing(casefolded)
        if show_steps:
            steps['tokenized'] = tokens
        
        # Step 4: Remove Stopwords
        no_stopwords = self.remove_stopwords(tokens)
        if show_steps:
            steps['no_stopwords'] = no_stopwords
        
        # Step 5: Normalization
        normalized = self.normalization(no_stopwords)
        if show_steps:
            steps['normalized'] = normalized
        
        # Step 6: Stemming
        stemmed = self.stemming(normalized)
        if show_steps:
            steps['stemmed'] = stemmed
        
        final_text = ' '.join(stemmed)
        if show_steps:
            steps['final'] = final_text
            return final_text, steps
        
        return final_text
    
    def load_and_preprocess_data(self, filepath):
        """Load dan preprocess dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(lambda x: self.preprocess_text(x))
        
        # Hapus baris dengan teks kosong setelah preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        return df
    
    def balance_dataset(self, df):
        """Menyeimbangkan dataset untuk training model"""
        print("Balancing dataset...")
        
        # Pisahkan berdasarkan kelas
        positive = df[df['sentiment'] == 1]
        negative = df[df['sentiment'] == 0]
        
        print(f"Original distribution - Positive: {len(positive)}, Negative: {len(negative)}")
        
        # Tentukan ukuran target (ambil yang terkecil atau rata-rata)
        min_size = min(len(positive), len(negative))
        
        # Resample untuk menyeimbangkan
        if len(positive) > min_size:
            positive_resampled = resample(positive, n_samples=min_size, random_state=42)
        else:
            positive_resampled = positive
            
        if len(negative) > min_size:
            negative_resampled = resample(negative, n_samples=min_size, random_state=42)
        else:
            negative_resampled = negative
        
        # Gabungkan dataset yang seimbang
        balanced_df = pd.concat([positive_resampled, negative_resampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced distribution - Positive: {len(positive_resampled)}, Negative: {len(negative_resampled)}")
        
        return balanced_df
    
    def train_model(self, df_balanced):
        """Training model SVM"""
        print("Training SVM model...")
        
        X = df_balanced['processed_text']
        y = df_balanced['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train SVM
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluasi
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))
        
        return accuracy
    
    def predict_sentiment(self, text):
        """Prediksi sentimen untuk teks baru"""
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'sentiment': 'Tidak dapat menentukan',
                'confidence': 0.0,
                'probability_negative': 0.5,
                'probability_positive': 0.5
            }
        
        text_tfidf = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        sentiment_label = "Positif" if prediction == 1 else "Negatif"
        confidence = max(probability)
        
        return {
            'sentiment': sentiment_label,
            'confidence': confidence,
            'probability_negative': probability[0],
            'probability_positive': probability[1]
        }
    
    def save_model(self, filepath='sentiment_model.pkl'):
        """Simpan model dan komponen lainnya"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'stemmer': self.stemmer,
            'stop_words': self.stop_words,
            'normalization_dict': self.normalization_dict
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='sentiment_model.pkl'):
        """Load model dan komponen lainnya"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.stemmer = model_data['stemmer']
        self.stop_words = model_data['stop_words']
        self.normalization_dict = model_data['normalization_dict']
        
        print(f"Model loaded from {filepath}")

def main():
    """Fungsi utama untuk training model"""
    analyzer = SentimentAnalyzer()
    
    # Load dan preprocess data asli (untuk dashboard)
    print("Processing original dataset for dashboard...")
    df_original = analyzer.load_and_preprocess_data('mbg.csv')
    df_original.to_csv('mbg_processed.csv', index=False)
    
    # Load dan preprocess data untuk training (dengan balancing)
    print("\nProcessing dataset for model training...")
    df_for_training = analyzer.load_and_preprocess_data('mbg.csv')
    df_balanced = analyzer.balance_dataset(df_for_training)
    
    # Training model
    accuracy = analyzer.train_model(df_balanced)
    
    # Simpan model
    analyzer.save_model('sentiment_model.pkl')
    
    # Simpan dataset balanced untuk referensi
    df_balanced.to_csv('mbg_balanced.csv', index=False)
    
    print("\nModel training completed!")
    print(f"Original dataset saved to: mbg_processed.csv")
    print(f"Balanced dataset saved to: mbg_balanced.csv")
    print(f"Model saved to: sentiment_model.pkl")

if __name__ == "__main__":
    main()
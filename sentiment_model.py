import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
        
        # SVM Model (Kernel RBF)
        self.model = SVC(kernel='rbf', C=1.0, gamma=1, probability=True)
        
        # Kamus normalisasi bahasa Indonesia
        self.normalization_dict = {
            'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'krna': 'karena',
            'tp': 'tapi', 'tpi': 'tapi', 'gk': 'tidak', 'ga': 'tidak',
            'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'g': 'tidak',
            'tdk': 'tidak', 'gitu': 'begitu', 'gt': 'begitu', 'gmn': 'bagaimana',
            'gimana': 'bagaimana', 'dmn': 'dimana',
            'kmn': 'kemana',
            'knp': 'kenapa', 'knapa': 'kenapa', 'org': 'orang', 'orng': 'orang',
            'tmn': 'teman', 'temen': 'teman', 'bgmn': 'bagaimana', 'bgt': 'banget',
            'banget': 'sangat', 'bener': 'benar', 'bnr': 'benar', 'bnyk': 'banyak',
            'bnyak': 'banyak', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah',
            'telah': 'sudah', 'blm': 'belum', 'blom': 'belum', 'msh': 'masih',
            'msih': 'masih', 'lg': 'lagi', 'lgi': 'lagi', 'skrg': 'sekarang',
            'skrang': 'sekarang', 'skg': 'sekarang', 'nanti': 'nanti',
            'ntar': 'nanti', 'tar': 'nanti', 'bsk': 'besok', 'besok': 'besok',
            'kmrn': 'kemarin', 'kmarin': 'kemarin', 'hrs': 'harus',
            'kudu': 'harus', 'mesti': 'harus', 'bs': 'bisa', 'bsa': 'bisa',
            'isa': 'bisa', 'biar': 'agar', 'spy': 'agar', 'supaya': 'agar',
            'kalo': 'kalau', 'klo': 'kalau', 'jd': 'jadi', 'jadi': 'menjadi',
            'jdnya': 'jadinya', 'jadinya': 'akhirnya', 'jg': 'juga', 'jga': 'juga',
            'jgn': 'jangan', 'jngn': 'jangan', 'jgn2': 'jangan-jangan',
            'aj': 'saja', 'aja': 'saja', 'doang': 'saja', 'aje': 'saja',
            'cm': 'cuma', 'cuma': 'hanya', 'cman': 'hanya', 'ckp': 'cukup',
            'cukup': 'cukup', 'krg': 'kurang', 'kurang': 'kurang', 'emg': 'memang',
            'emang': 'memang', 'mmg': 'memang', 'sbnrnya': 'sebenarnya',
            'sbenernya': 'sebenarnya', 'pdhl': 'padahal', 'pdahal': 'padahal',
            'wlpn': 'walaupun', 'walaupun': 'walaupun', 'meskipun': 'walaupun',
            'walau': 'walaupun', 'aplg': 'apalagi', 'apalagi': 'apalagi',
            'mgkn': 'mungkin', 'mungkin': 'mungkin', 'mgkin': 'mungkin',
            'kyknya': 'kayaknya', 'kyaknya': 'kayaknya', 'kayaknya': 'sepertinya',
            'kyk': 'seperti', 'kayak': 'seperti', 'ky': 'seperti', 'sprt': 'seperti',
            'kaya': 'seperti', 'sy': 'saya', 'gw': 'saya', 'gue': 'saya',
            'gua': 'saya', 'w': 'saya', 'aku': 'saya', 'ak': 'saya', 'km': 'kamu',
            'kmu': 'kamu', 'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu', 'elo': 'kamu',
            'u': 'kamu', 'dy': 'dia', 'dia': 'dia', 'mrk': 'mereka',
            'mreka': 'mereka', 'tololl': 'bodoh', 'tolol': 'bodoh',
            'qt': 'kita', 'qta': 'kita', 'seneng': 'senang', 'suka': 'suka',
            'sk': 'suka', 'kesel': 'kesal', 'binun': 'bingung', 'males': 'malas',
            'capek': 'capek', 'cape': 'capek', 'lelah': 'lelah', 'tired': 'lelah',
            'stress': 'stres', 'mantul': 'mantap', 'keren': 'keren', 'gokil': 'keren',
            'ajib': 'keren', 'top': 'bagus', 'the best': 'terbaik',
            'terbaik': 'terbaik', 'terburuk': 'terburuk', 'worst': 'terburuk',
            'best': 'terbaik', 'good': 'bagus', 'bad': 'buruk', 'nice': 'bagus',
            'awesome': 'keren', 'amazing': 'menakjubkan', 'terrible': 'buruk',
            'horrible': 'mengerikan', 'excellent': 'sangat bagus', 'perfect': 'sempurna',
            'ok': 'baik', 'oke': 'baik', 'okay': 'baik', 'fine': 'baik', 'standard': 'standar',
            'ajg':'anjing', 'anjg':'anjing', 'tw':'tau', 'kek':'seperti'
        }
        
    def text_cleaning(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def case_folding(self, text):
        return text.lower()
    
    def tokenizing(self, text):
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 1 and token.isalpha()]
        return tokens
    
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def normalization(self, tokens):
        normalized_tokens = []
        for token in tokens:
            if token in self.normalization_dict:
                normalized_tokens.append(self.normalization_dict[token])
            else:
                normalized_tokens.append(token)
        return normalized_tokens
    
    def stemming(self, tokens):
        text = ' '.join(tokens)
        stemmed_text = self.stemmer.stem(text)
        return stemmed_text.split()
    
    def preprocess_text(self, text, show_steps=False):
        """
        Preprocessing dengan urutan:
        1. Cleaning
        2. Case Folding
        3. Tokenizing
        4. Normalization
        5. Stopwords Removal
        6. Stemming
        """
        steps = {}
        # Step 1: Cleaning
        cleaned = self.text_cleaning(text)
        if show_steps: steps['cleaned'] = cleaned
        # Step 2: Case Folding
        casefolded = self.case_folding(cleaned)
        if show_steps: steps['casefolded'] = casefolded
        # Step 3: Tokenizing
        tokens = self.tokenizing(casefolded)
        if show_steps: steps['tokenized'] = tokens
        # Step 4: Normalization
        normalized = self.normalization(tokens)
        if show_steps: steps['normalized'] = normalized
        # Step 5: Remove Stopwords
        no_stopwords = self.remove_stopwords(normalized)
        if show_steps: steps['no_stopwords'] = no_stopwords
        # Step 6: Stemming
        stemmed = self.stemming(no_stopwords)
        if show_steps: steps['stemmed'] = stemmed
        final_text = ' '.join(stemmed)
        if show_steps:
            steps['original'] = text
            steps['final'] = final_text
            return final_text, steps
        return final_text
    
    def load_and_preprocess_data(self, filepath):
        print(f"Loading dataset from {filepath}...")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠ UTF-8 gagal, mencoba encoding latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')
        
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(lambda x: self.preprocess_text(x))
        df = df[df['processed_text'].str.len() > 0]
        df['sentiment'] = df['sentiment'].astype(int)
        
        print("Preprocessing complete.")
        return df
    

    def print_confusion_matrix(self, y_test, y_pred, title="Confusion Matrix"):
        """
        Menampilkan confusion matrix dengan format yang jelas
        """
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{title}")
        print("="*60)
        print(f"\nDetail Metrik dari Confusion Matrix:")
        print(f" * True Positive (TP)  : {tp:<5} (Prediksi: Positif, Aktual: Positif)")
        print(f" * True Negative (TN)  : {tn:<5} (Prediksi: Negatif, Aktual: Negatif)")
        print(f" * False Positive (FP) : {fp:<5} (Prediksi: Positif, Aktual: Negatif) -> Error Tipe I")
        print(f" * False Negative (FN) : {fn:<5} (Prediksi: Negatif, Aktual: Positif) -> Error Tipe II")
        
        print("\nMatriks Konfusi (Visual):")
        print("                     Prediksi Negatif | Prediksi Positif")
        print("---------------------------------------------------------")
        print(f"Aktual Negatif (0) |     {tn:<10} |     {fp:<10}")
        print(f"Aktual Positif (1) |     {fn:<10} |     {tp:<10}")
        print("---------------------------------------------------------")
    
    def train_and_evaluate_model(self, df):
        """
        Training model dan evaluasi performa
        """
        print("\n" + "="*60)
        print("TRAINING MODEL DENGAN DATA ORIGINAL")
        print("="*60)
        
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train SVM
        print("Training SVM model...")
        self.model.fit(X_train_tfidf, y_train)
        print("✓ Training selesai!")
        
        # Evaluasi Model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        self.print_confusion_matrix(y_test, y_pred, "Confusion Matrix")
        
        return accuracy

    def predict_sentiment(self, text):
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
        model_data = {
            'model': self.model, 
            'vectorizer': self.vectorizer,
            'stemmer': self.stemmer, 
            'stop_words': self.stop_words,
            'normalization_dict': self.normalization_dict
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='sentiment_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.stemmer = model_data['stemmer']
        self.stop_words = model_data['stop_words']
        self.normalization_dict = model_data['normalization_dict']
        print(f"✓ Model loaded from {filepath}")

def main():
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS - MULTI-RATIO PERFORMANCE TEST")
    print("="*60)
    
    analyzer = SentimentAnalyzer()

    # 1. Load Data
    df = analyzer.load_and_preprocess_data('data_mbg_labelled.csv')
    
    # Save processed data for dashboard
    print("\nSaving processed data to mbg_processed.csv...")
    df.to_csv('mbg_processed.csv', index=False, encoding='utf-8')
    print("✓ Processed data saved successfully!")
    
    # List rasio yang akan diuji (test_size adalah kebalikannya)
    # 0.1 = 90:10, 0.2 = 80:20, 0.3 = 70:30
    test_ratios = [0.1, 0.2, 0.3]
    results = []
    best_accuracy = 0
    best_ratio = None
    best_test_size = None

    for test_size in test_ratios:
        ratio_name = f"{int((1-test_size)*100)}:{int(test_size*100)}"
        print("\n\n" + "#"*70)
        print(f" PENGUJIAN RASIO DATA {ratio_name}")
        print("#"*70)
        
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data dengan rasio saat ini
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Jumlah Data Training: {len(X_train)}")
        print(f"Jumlah Data Testing : {len(X_test)}")
        
        # TF-IDF
        X_train_tfidf = analyzer.vectorizer.fit_transform(X_train)
        X_test_tfidf = analyzer.vectorizer.transform(X_test)
        
        # Training
        analyzer.model.fit(X_train_tfidf, y_train)
        
        # Prediksi
        y_pred = analyzer.model.predict(X_test_tfidf)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Tampilkan Confusion Matrix Visual
        print(f"\nConfusion Matrix ({ratio_name}):")
        print(f"{'':<15} | Pred Negatif | Pred Positif")
        print("-" * 45)
        print(f"{'Aktual Negatif':<15} | {tn:<12} | {fp:<12}")
        print(f"{'Aktual Positif':<15} | {fn:<12} | {tp:<12}")
        
        # Tampilkan Classification Report
        print(f"\nClassification Report ({ratio_name}):")
        print(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))
        
        # Simpan ringkasan untuk tabel akhir
        results.append({
            'Rasio': ratio_name,
            'Accuracy': acc,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })
        
        # Cek apakah ini rasio terbaik
        if acc > best_accuracy:
            best_accuracy = acc
            best_ratio = ratio_name
            best_test_size = test_size

    # 2. Ringkasan Akhir untuk Tabel Skripsi
    print("\n\n" + "="*70)
    print("RINGKASAN PERFORMA UNTUK TABEL SKRIPSI")
    print("="*70)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("="*70)
    
    # 3. Tampilkan Rasio Terbaik
    print("\n" + "="*70)
    print("RASIO TERBAIK BERDASARKAN ACCURACY")
    print("="*70)
    print(f"Rasio Terbaik: {best_ratio}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("="*70)

    # 4. Train ulang model dengan rasio terbaik dan simpan
    print(f"\n🔄 Training ulang model dengan rasio terbaik ({best_ratio})...")
    X = df['processed_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=best_test_size, random_state=42, stratify=y
    )
    # TF-IDF
    X_train_tfidf = analyzer.vectorizer.fit_transform(X_train)
    X_test_tfidf = analyzer.vectorizer.transform(X_test)
    # Training
    analyzer.model.fit(X_train_tfidf, y_train)
    
    # Calculate performance metrics for the best model
    y_pred = analyzer.model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)
    
    # Save performance metrics to JSON
    import json
    metrics = {
        'best_ratio': best_ratio,
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'true_negative': int(cm[0][0]),
            'false_positive': int(cm[0][1]),
            'false_negative': int(cm[1][0]),
            'true_positive': int(cm[1][1])
        },
        'classification_report': {
            'negative': {
                'precision': float(report['Negatif']['precision']),
                'recall': float(report['Negatif']['recall']),
                'f1-score': float(report['Negatif']['f1-score'])
            },
            'positive': {
                'precision': float(report['Positif']['precision']),
                'recall': float(report['Positif']['recall']),
                'f1-score': float(report['Positif']['f1-score'])
            },
            'weighted_avg': {
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'f1-score': float(report['weighted avg']['f1-score'])
            }
        }
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("✓ Model metrics saved to model_metrics.json")
    
    # Simpan model dengan rasio terbaik
    analyzer.save_model('sentiment_model.pkl')
    print(f"✓ Model dengan rasio {best_ratio} berhasil disimpan!")

if __name__ == "__main__":
    main()
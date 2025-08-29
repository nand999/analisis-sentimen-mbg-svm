import subprocess
import sys
import os

def install_requirements():
    """Install semua requirements"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_model_training():
    """Jalankan training model"""
    print("Training sentiment analysis model...")
    subprocess.check_call([sys.executable, "sentiment_model.py"])

def run_dashboard():
    """Jalankan dashboard Streamlit"""
    print("Starting Streamlit dashboard...")
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

def main():
    """Fungsi utama"""
    print("=== Sentiment Analysis Pipeline ===")
    
    # Cek apakah file dataset ada
    if not os.path.exists("mbg.csv"):
        print("Error: File mbg.csv tidak ditemukan!")
        print("Pastikan file dataset ada di direktori yang sama.")
        return
    
    try:
        # Install requirements
        install_requirements()
        
        # Training model
        run_model_training()
        
        # Jalankan dashboard
        run_dashboard()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
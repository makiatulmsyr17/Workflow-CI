# modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings

# Abaikan peringatan
warnings.filterwarnings("ignore")

# --- 1. Memuat & Persiapan Data ---
try:
    # Ganti dengan nama file data Anda yang sudah bersih
    df = pd.read_csv('heart_preprocessing.csv')
except FileNotFoundError:
    print("File 'heart_preprocess.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

X = df.drop('target', axis=1)
y = df['target']

# Split data dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Pelatihan Model dengan MLflow Autolog ---

# Atur nama eksperimen
mlflow.set_experiment("Eksperimen Penyakit Jantung (Basic)")

# Aktifkan autologging untuk Scikit-learn
# Ini akan secara otomatis mencatat parameter, metrik, dan model
mlflow.sklearn.autolog()

# Mulai sesi MLflow
with mlflow.start_run(run_name="RandomForest_Autolog"):
    
    # Definisikan dan latih model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Autolog akan otomatis mencatat metrik saat evaluasi
    # (tidak perlu kode evaluasi eksplisit di dalam run jika hanya untuk logging)
    print("Model dilatih dengan autologging.")
    
print("Eksperimen (Basic) selesai. Cek folder 'mlruns' atau jalankan 'mlflow ui' di terminal.")
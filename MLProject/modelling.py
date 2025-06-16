# modelling.py (Diperbarui untuk MLflow Project)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

# --- 1. Memuat & Persiapan Data ---
try:
    df = pd.read_csv('heart_preprocess.csv')
except FileNotFoundError:
    print("File 'heart_preprocess.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Pelatihan Model dengan Autolog ---
# Karena skrip ini dijalankan oleh 'mlflow run', sebuah run sudah aktif.
# Kita hanya perlu mengaktifkan autologging.
print("Mengaktifkan MLflow autolog...")
mlflow.sklearn.autolog()

# Definisikan dan latih model.
# Autolog akan otomatis mencatat parameter, metrik, dan model ke dalam run yang sudah aktif.
print("Melatih model RandomForestClassifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluasi model untuk memicu pencatatan metrik oleh autolog
score = model.score(X_test_scaled, y_test)

print(f"Pelatihan model selesai. Skor akurasi: {score:.4f}")
print("Logging ke MLflow selesai.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import warnings

# Mengabaikan peringatan
warnings.filterwarnings("ignore")

print("--- Memulai skrip pembuatan model ---")

# 1. Memuat Dataset
# Pastikan file 'heart_preprocessing.csv' ada di dalam folder 'MLProject'
try:
    df = pd.read_csv("Workflow-CI\\MLProject\\heart_preprocessing.csv")
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: File 'heart_preprocessing.csv' tidak ditemukan di dalam folder 'MLProject'.")
    print("Pastikan Anda sudah meletakkannya di sana.")
    exit()

# 2. Persiapan Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data berhasil disiapkan.")

# 3. Menggunakan MLflow Autolog
# Ini cara paling simpel untuk memastikan semuanya tercatat
mlflow.sklearn.autolog()

# 4. Melatih Model
with mlflow.start_run() as run:
    print("Memulai pelatihan model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")

    # Evaluasi sederhana untuk memicu logging
    accuracy = model.score(X_test, y_test)
    print(f"Akurasi model: {accuracy}")
    print(f"Model dan metrik berhasil disimpan di folder 'mlruns' dengan Run ID: {run.info.run_id}")

print("--- Skrip selesai ---")

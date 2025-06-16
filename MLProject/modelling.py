# modelling.py (Diperbarui untuk menemukan path secara otomatis)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings
import os # Tambahkan import ini

warnings.filterwarnings("ignore")

# --- 1. Memuat Data dengan Path yang Robust ---
# Dapatkan path direktori tempat skrip ini berada
script_dir = os.path.dirname(__file__)
# Gabungkan path tersebut dengan nama file data
data_path = os.path.join(script_dir, 'Workflow-CI', 'MLProject', 'heart_preprocessing.csv')

try:
    df = pd.read_csv(data_path)
    print(f"Dataset berhasil dimuat dari: {data_path}")
except FileNotFoundError:
    print(f"File dataset tidak ditemukan di path: {data_path}")
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
print("Mengaktifkan MLflow autolog...")
mlflow.sklearn.autolog()

print("Melatih model RandomForestClassifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluasi
score = model.score(X_test_scaled, y_test)
print(f"Pelatihan model selesai. Skor akurasi: {score:.4f}")
print("Logging ke MLflow selesai.")
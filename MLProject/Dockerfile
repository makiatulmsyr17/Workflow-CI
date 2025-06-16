# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .

# Install dependencies menggunakan pip
# Cara ini lebih robust untuk instalasi di dalam Docker
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn

# Perintah default untuk menjalankan proyek saat container dimulai
ENTRYPOINT ["mlflow", "run", "."]
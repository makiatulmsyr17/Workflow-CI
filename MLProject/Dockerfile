# Gunakan base image Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file environment dan install dependencies
COPY conda.yaml .
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir -r <(grep -E '^[a-zA-Z]' conda.yaml | sed 's/- //')

# Salin semua file proyek ke dalam container
COPY . .

# Perintah default untuk menjalankan proyek saat container dimulai
ENTRYPOINT ["mlflow", "run", "."]

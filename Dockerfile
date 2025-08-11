# Gunakan image Python resmi sebagai base image
FROM python:3.10-slim-buster

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu
# Ini akan memanfaatkan caching Docker jika file tidak berubah
COPY requirements.txt .

# Install dependencies tanpa menghapus cache pip
RUN pip install --no-cache-dir -r requirements.txt

# --- NLTK Data Download ---
# Unduh data NLTK yang dibutuhkan oleh preprocessor Anda
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')"
ENV NLTK_DATA=/usr/local/nltk_data

# Salin folder 'app' yang berisi semua .py dan folder lainnya
COPY ./app /app/app

# Salin folder 'model/optimized' secara terpisah supaya model ikut ke image
COPY ./app/model/optimized /app/model/optimized

# Salin folder 'static' untuk front-end atau file statis lainnya
COPY ./static /app/static

# Buat direktori untuk logs, results, uploads, plots
RUN mkdir -p /app/logs /app/results /app/uploads /app/plots

# Buka port 8000 untuk akses FastAPI (default Uvicorn)
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
# --factory digunakan jika Anda menggunakan fungsi create_app()
CMD ["uvicorn", "app.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
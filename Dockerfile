# 1️⃣ Use a Python base image
FROM python:3.11-slim

# 2️⃣ Install system dependencies (libGL + libglib needed for OpenCV/DeepFace)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# 3️⃣ Set working directory
WORKDIR /app

# 4️⃣ Copy dependencies
COPY requirements.txt .

# 5️⃣ Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy the entire project
COPY . .

# 7️⃣ Expose Flask port
EXPOSE 5000

# 8️⃣ Run Flask automatically
CMD ["python", "app/app.py"]

FROM python:3.11-slim

WORKDIR /app

# Установка минимальных системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt первым для лучшего кэширования слоев
COPY requirements.txt .

# Устанавливаем зависимости (CPU-only PyTorch для быстрой сборки)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Создаем папки для моделей и данных
RUN mkdir -p /app/models /app/data /app/uploads

# Открываем порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
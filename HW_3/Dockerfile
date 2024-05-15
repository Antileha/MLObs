FROM python:3.8

# Установка зависимостей для создания виртуального окружения
RUN python -m venv /opt/venv

# Активация виртуального окружения
ENV PATH="/opt/venv/bin:$PATH"

# Установка зависимостей
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# Копирование файлов проекта
COPY . /app
WORKDIR /app

# Обучение модели
RUN python /app/src/train_model.py

# Запуск Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
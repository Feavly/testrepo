# Используем образ с Python
FROM python:3

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем исходный код в контейнер
COPY Lab2.py /app.Lab2.py

# Запускаем приложение при старте контейнера
CMD ["python", "/app/Lab2/py"]
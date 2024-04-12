from sklearn.linear_model import LinearRegression
import numpy as np
import os
import joblib

# Функция для загрузки данных из CSV файла
def load_data(file_path):
    """
    Загружает данные из CSV файла.

    Параметры:
        file_path (str): Путь к CSV файлу.

    Возвращает:
        X (numpy.ndarray): Матрица признаков.
        y (numpy.ndarray): Массив меток.
    """
    data = np.loadtxt(file_path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

# Функция для обучения модели линейной регрессии
def train_model(X, y):
    """
    Обучает модель линейной регрессии.

    Параметры:
        X (numpy.ndarray): Матрица признаков.
        y (numpy.ndarray): Массив меток.

    Возвращает:
        model: Обученная модель линейной регрессии.
    """
    model = LinearRegression()  # Создание модели линейной регрессии
    model.fit(X.reshape(-1, 1), y)  # Обучение модели на данных
    return model

# Функция для сохранения обученной модели в файл
def save_model(model, directory, filename):
    """
    Сохраняет обученную модель в файл.

    Параметры:
        model: Обученная модель.
        directory (str): Директория для сохранения файла.
        filename (str): Имя файла.

    Возвращает:
        None
    """
    if not os.path.exists(directory):  # Проверка существования директории
        os.makedirs(directory)  # Создание директории, если она не существует
    joblib.dump(model, os.path.join(directory, filename))  # Сохранение модели в файл

# Главная функция
def main():
    # Загрузка предварительно обработанных тренировочных данных
    X_train_scaled = np.loadtxt('train/preprocessed_data_train.csv', delimiter=',')

    # Загрузка меток тренировочных данных
    _, y_train = load_data('train/data_train.csv')

    # Обучение модели
    model = train_model(X_train_scaled, y_train)

    # Сохранение обученной модели
    save_model(model, 'models', 'trained_model.pkn')

if __name__ == "__main__":
    main()
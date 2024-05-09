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

# Функция для загрузки модели из файла
def load_model(file_path):
    """
    Загружает модель из файла.

    Параметры:
        file_path (str): Путь к файлу с моделью.

    Возвращает:
        model: Загруженная модель.
    """
    return joblib.load(file_path)

# Функция для оценки модели на тестовых данных
def evaluate_model(model, X, y):
    """
    Оценивает модель на тестовых данных.

    Параметры:
        model: Обученная модель.
        X (numpy.ndarray): Массив признаков.
        y (numpy.ndarray): Массив целевых значений.

    Возвращает:
        float: Среднеквадратичное отклонение модели на тестовых данных.
    """
    # Решейпим X, чтобы получить 2D массив с одним столбцом
    X_reshaped = X.reshape(-1, 1)
    predictions = model.predict(X_reshaped)
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    return rmse

def main():
    # Загрузка предварительно обработанных тестовых данных
    X_test_scaled = np.loadtxt('train/preprocessed_data_train.csv', delimiter=',')

    # Загрузка меток тестовых данных
    _, y_test = load_data('train/data_train.csv')

    # Загрузка обученной модели
    model = load_model('models/trained_model.npy')

    # Оценка модели
    rmse = evaluate_model(model, X_test_scaled, y_test)
    print("Root Mean Squared Error (RMSE) on test data:", rmse)

if __name__ == "__main__":
    main()
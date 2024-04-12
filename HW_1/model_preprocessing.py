from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Функция для загрузки данных из файла
def load_data(file_path):
    """
    Загружает данные из CSV файла.

    Параметры:
        file_path (str): Путь к CSV файлу.

    Возвращает:
        X (numpy.ndarray): Матрица признаков.
        y (numpy.ndarray): Массив меток.
    """
    data = np.loadtxt(file_path, delimiter=',')  # Загрузка данных из CSV файла
    X, y = data[:, :-1], data[:, -1]  # Разделение признаков (X) и меток (y)
    return X, y

# Функция для предварительной обработки данных
def preprocess_data(X):
    """
    Предварительно обрабатывает данные, масштабируя их с помощью StandardScaler.

    Параметры:
        X (numpy.ndarray): Матрица признаков.

    Возвращает:
        X_scaled (numpy.ndarray): Масштабированная матрица признаков.
    """
    scaler = StandardScaler()  # Создание объекта StandardScaler
    X_scaled = scaler.fit_transform(X)  # Масштабирование данных
    return X_scaled

# Функция для сохранения предварительно обработанных данных в файл
def save_preprocessed_data(X, directory, filename):
    """
    Сохраняет предварительно обработанные данные в CSV файл.

    Параметры:
        X (numpy.ndarray): Матрица признаков.
        directory (str): Директория для сохранения файла.
        filename (str): Имя файла.
    """
    if not os.path.exists(directory):  # Проверка существования директории
        os.makedirs(directory)  # Создание директории, если она не существует
    np.savetxt(os.path.join(directory, filename), X, delimiter=',')  # Сохранение данных в CSV файл

# Главная функция
def main():
    # Загрузка тренировочных данных
    X_train, y_train = load_data('train/data_train.csv')

    # Предварительная обработка тренировочных данных
    X_train_scaled = preprocess_data(X_train)

    # Сохранение предварительно обработанных тренировочных данных
    save_preprocessed_data(X_train_scaled, 'train', 'preprocessed_data_train.csv')

if __name__ == "__main__":
    main()
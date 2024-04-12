import numpy as np
import os

# Функция для генерации синтетических данных
def generate_data(n_samples, noise_level=0.1):
    # Генерация равномерно распределенных образцов между 0 и 10
    X = np.linspace(0, 10, n_samples)
    # Генерация значений y с использованием синусоидальной функции с добавлением шума
    y = np.sin(X) + np.random.normal(scale=noise_level, size=n_samples)
    return X, y

# Функция для добавления аномалий в данные
def add_anomaly(X, y, anomaly_idx, anomaly_value):
    # Добавление аномалии к значению y в указанном индексе
    y[anomaly_idx] += anomaly_value
    return X, y

# Функция для сохранения данных в CSV файл
def save_data(X, y, directory, filename):
    # Создание директории, если она не существует
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Сохранение значений X и y как столбцов в CSV файле
    np.savetxt(os.path.join(directory, filename), np.column_stack((X, y)), delimiter=',')

# Главная функция
def main():
    # Генерация и сохранение тренировочных данных
    X_train, y_train = generate_data(100)
    save_data(X_train, y_train, 'train', 'data_train.csv')

    # Генерация и сохранение тестовых данных с аномалиями
    X_test, y_test = generate_data(50)
    # Добавление аномалии к тестовым данным в индексе 25 со значением 2.0
    X_test, y_test = add_anomaly(X_test, y_test, 25, 2.0)
    save_data(X_test, y_test, 'test', 'data_test_anomaly.csv')

# Точка входа в скрипт
if __name__ == "__main__":
    main()
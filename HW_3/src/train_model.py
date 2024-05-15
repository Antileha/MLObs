import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Константы
RAND = 42
MODEL_PATH = '/app/data/iris_model.pkl'

# Загрузка датасета Iris
data = load_iris()
X, y = data['data'], data['target']

# Разбиение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RAND)

# Обучение модели RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=RAND)
model.fit(X_train, y_train)

# Проверка существования директории
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Сохранение обученной модели
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f'Модель сохранена в {MODEL_PATH}')
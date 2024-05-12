import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RAND = 42


def train_model():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    model = RandomForestClassifier(n_estimators=100, random_state=RAND)
    model.fit(X_train, y_train.values.ravel())

    # Сохранение модели
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("Модель обучена и сохранена как iris_model.pkl")


if __name__ == "__main__":
    train_model()
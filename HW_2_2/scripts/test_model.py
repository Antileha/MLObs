import pickle
from sklearn.metrics import accuracy_score
import pandas as pd


def test_model():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    # Загрузка модели

    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    test_model()
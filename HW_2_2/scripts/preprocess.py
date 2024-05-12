import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv('iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    for name, dataset in datasets.items():
        dataset.to_csv(f"{name}.csv", index=False)
    print("Данные предобработаны и сохранены в наборы для обучения и тестирования.")

if __name__ == "__main__":
    preprocess()
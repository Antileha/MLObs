import pandas as pd
from sklearn import datasets
# iris = datasets.load_iris()
# iris.keys()


def download_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('iris.csv', index=False)
    print('Data downloaded and saved to iris.csv')

if __name__ == '__main__':
    download_data()
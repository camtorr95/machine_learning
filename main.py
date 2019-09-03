import pandas as pd
from sklearn import preprocessing
from models import *


def process_data(x):
    x = _minmax_scaler.fit_transform(x)
    x = _std_scaler.fit_transform(x)
    x = np.c_[np.ones(x.shape[0]).reshape(x.shape[0], 1), x]
    return x


def regmodel_init(target, csv_file):
    df = pd.read_csv(csv_file)
    x = df.drop(target, axis=1).values
    x = process_data(x)
    y = df[target].values.reshape(x.shape[0], 1)
    return x, y


def main():
    x, y = regmodel_init(target="final_exam", csv_file="exams.csv")
    # _x = np.array([1, 852, 2, 1, 36]).reshape(5, 1)
    model = linear_regression(x, y)
    model.train()
    print(x)
    print(model.theta)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    _std_scaler = preprocessing.StandardScaler()
    _minmax_scaler = preprocessing.MinMaxScaler()
    main()

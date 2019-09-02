import numpy as np
import pandas as pd
from sklearn import preprocessing
from models import *


def regmodel_init(target, csv_file):
    df = pd.read_csv(csv_file)
    df.insert(0, 'ones', 1)
    x = _scaler.fit_transform(df.drop(target, axis=1).values)
    y = df[target].values.reshape(x.shape[0], 1)
    return x, y


def main():
    x, y = regmodel_init(target="sold", csv_file="housing.csv")
    _x = np.array([1, 2104, 5, 1, 45, 460]).reshape(6, 1)
    _x = _scaler.fit_transform(_x)
    model = oneclass_logistic_regression(x, y)
    model.train()
    print(model.apply(_x))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    _scaler = preprocessing.StandardScaler()
    main()

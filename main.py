import pandas as pd
import matplotlib.pyplot as plt
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
    x, y = regmodel_init(target="sold", csv_file="housing.csv")
    _x = np.array([1, 852, 2, 1, 36, 178]).reshape(6, 1)
    alpha = 1
    niterations = 4000
    atol = 0.001
    model = oneclass_logistic_regression(x, y, alpha=alpha, niterations=niterations, atol=atol)
    model.train()
    print("Convergence" + " atol=" + str(atol) + ": " + str(model.gd[1]))
    # plt.plot(model.gd[2].reshape(niterations, 1))
    # plt.show()
    print(model.apply(_x))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    _std_scaler = preprocessing.StandardScaler()
    _minmax_scaler = preprocessing.MinMaxScaler()
    main()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from models import *


def process_data(x):
    # x = _minmax_scaler.fit_transform(x)
    # x = _std_scaler.fit_transform(x)
    x = np.c_[np.ones(x.shape[0]).reshape(x.shape[0], 1), x]
    return x


def regmodel_init(target, csv_file):
    df = pd.read_csv(csv_file)
    xdf = df.drop(target, axis=1)
    x = process_data(xdf.values)
    y = df[target].values.reshape(x.shape[0], 1)
    return x, y, xdf


def main():
    x, y, xdf = regmodel_init(target="y", csv_file="ex1/ex1data1.csv")

    lin_reg = linear_regression(x, y, alpha=0.01, niterations=1500)
    lin_reg.gradient_descent_opt()
    _x = np.array([1, 7]).reshape(2, 1)
    # print(lin_reg.gd[2][0])
    # print(_x)
    # print(lin_reg.apply(_x))
    theta = lin_reg.theta

    def apply_theta(xi):
        return theta[0] + theta[1] * xi

    _at = np.vectorize(apply_theta)
    _xi = np.arange(xdf.min().values, xdf.max().values)
    print(_xi)
    print(_at(_xi))

    plt.scatter(xdf, y, c='r', marker='o')
    plt.plot(_xi, _at(_xi))
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    _std_scaler = preprocessing.StandardScaler()
    _minmax_scaler = preprocessing.MinMaxScaler()
    main()

"""
Pasos Para aplicar un modelo de regresión

1. Identificar las variables x, y, theta, m, alpha
2. Seleccionar una función de costo, e.g. squuared_error
3. Minimizar la función de costo con respecto a theta (minimos locales, globales)
3.1.1 Definir el gradiente (derivada con respecto a theta) de la función de costo
3.1.2 Aplicar algoritmo de gradient_descent u otro algoritmo para encontrar minimos en funciones multivariables
3.2.1 Aplicar función normal
4. Aplicar theta * x para un nuevo dato x
-------
Si n (features, características o atributos)
    n >= 10000,
Usar gradient_descent, de lo contrario usar normal_equation

NON INVERTIBLE MATRICES
    - Redundant feautes (linearly dependent)
    - Too many features (m <= n)

Usar regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def linear_hypothesis(x, theta):
    return theta.T @ x


def logistic_hypothesis(x, theta):
    return ((theta.T @ x) >= 0.5).astype(int)


def logistic_cost_function(x, y, m, theta):
    h = sigmoid(x @ theta)
    cost = -y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)
    return cost / m


def logistic_gradient(x, y, m, theta):
    h = sigmoid(x @ theta)
    return (x.T @ (h - y)) / m


def normal_equation(x, y):
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def mean_squared_error_gradient(x, y, m, theta):
    return x.T @ (x @ theta - y) / m


def mean_squared_error_cost(x, y, m, theta):
    return np.square(x @ theta - y) / (2 * m)


def gradient_descent(niterations, alpha, x, y, m, n, theta, gradient):
    step = np.zeros(shape=(n, 1))
    zeros = np.zeros(shape=(n, 1))

    cost_function_by_niter = []

    for _ in range(niterations):
        cost_function_by_niter.append(np.sum(mean_squared_error_cost(x, y, m, theta)))
        step = alpha * gradient(x, y, m, theta)
        theta = theta - step
    convergence = np.allclose(step, zeros)
    return theta, convergence, np.array(cost_function_by_niter)


def init():
    target = 'sold'
    df = pd.read_csv("housing.csv")
    df.insert(0, 'ones', 1)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(df.drop(target, axis=1).values)
    m, n = x.shape
    y = df[target].values.reshape(m, 1)
    return x, y, m, n


def main():
    x, y, m, n = init()
    theta = np.zeros(shape=(n, 1))
    alpha = 0.3
    theta = gradient_descent(10000, alpha, x, y, m, n, theta, logistic_gradient)
    _x = np.array([1, 1534, 3, 2, 30, 315]).reshape(n, 1)
    print(logistic_hypothesis(_x, theta[0]))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()

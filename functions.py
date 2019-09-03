import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def normal_equation(x, y):
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def mean_squared_error_gradient(x, y, m, theta):
    return x.T @ (x @ theta - y) / m


def mean_squared_error_cost(x, y, m, theta):
    return np.square(x @ theta - y) / (2 * m)


def oneclass_logistic_function_cost(x, y, m, theta):
    h = sigmoid(x @ theta)
    cost = -y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)
    return cost / m


def one_class_logistic_function_gradient(x, y, m, theta):
    h = sigmoid(x @ theta)
    return (x.T @ (h - y)) / m


def gradient_descent(alpha, x, y, m, n, theta, cost, gradient, niterations=1000, atol=0.000001):
    step = np.zeros(shape=(n, 1))
    zeros = np.zeros(shape=(n, 1))
    # cost_function_table_by_niter = []

    for _ in range(niterations):
        # cost_function_table_by_niter.append(np.sum(cost(x, y, m, theta)))
        step = alpha * gradient(x, y, m, theta)
        theta = theta - step

    convergence = np.allclose(step, zeros, atol=atol)
    return theta, convergence

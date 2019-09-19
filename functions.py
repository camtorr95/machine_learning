import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def normal_equation(x, y):
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def mean_squared_error_gradient(x, y, theta):
    m, n = x.shape
    return x.T @ (x @ theta - y) / m


def mean_squared_error_cost(x, y, theta):
    m, n = x.shape
    return np.sum(np.square(x @ theta - y)) / (2 * m)


def oneclass_logistic_function_cost(x, y, theta):
    m, n = x.shape
    h = sigmoid(x @ theta)
    cost = y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)
    return -cost / m


def one_class_logistic_function_gradient(x, y, theta):
    m, n = x.shape
    h = sigmoid(x @ theta)
    return (x.T @ (h - y)) / m


def gradient_descent(alpha, x, y, theta, cost, gradient, niterations, atol):
    m, n = x.shape
    step = np.zeros(shape=(n, 1))
    zeros = np.zeros(shape=(n, 1))
    cost_function_by_niter = []

    for _ in range(niterations):
        cost_function_by_niter.append(cost(x, y, theta))
        step = alpha * gradient(x, y, theta)
        theta = theta - step

    convergence = np.allclose(step, zeros, atol=atol)
    return theta, convergence, np.array(cost_function_by_niter)

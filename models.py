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
from functions import mean_squared_error_cost
from functions import mean_squared_error_gradient
from functions import normal_equation
from functions import gradient_descent
from functions import oneclass_logistic_function_cost
from functions import one_class_logistic_function_gradient
from functions import sigmoid


class linear_regression:
    def __init__(self, x, y, alpha=0.5, cost=mean_squared_error_cost,
                 gradient=mean_squared_error_gradient, niterations=10000):
        self.x, self.y = x, y
        self.m, self.n = x.shape
        self.alpha = alpha
        self.niterations = niterations
        self.theta = np.zeros(self.n).reshape(self.n, 1)
        self.default_cost = cost
        self.default_gradient = gradient
        self.convergence = False

    def train(self):
        if self.n < 10000:
            self.theta = normal_equation(self.x, self.y)
        else:
            tt = gradient_descent(self.alpha, self.x, self.y, self.m, self.n, self.theta,
                                  self.default_cost, self.default_gradient, niterations=self.niterations)
            self.theta = tt[0]
            self.convergence = tt[1]

    def apply(self, x):
        return self.theta.T @ x


class oneclass_logistic_regression:
    def __init__(self, x, y, alpha=0.1, cost=oneclass_logistic_function_cost,
                 gradient=one_class_logistic_function_gradient, niterations=1000):
        self.x, self.y = x, y
        self.m, self.n = x.shape
        self.alpha = alpha
        self.niterations = niterations
        self.theta = np.zeros(self.n).reshape(self.n, 1)
        self.default_cost = cost
        self.default_gradient = gradient

    def train(self):
        if self.n < 10000:
            self.theta = normal_equation(self.x, self.y)
        else:
            self.theta = gradient_descent(self.alpha, self.x, self.y, self.m, self.n, self.theta,
                                          self.default_cost, self.default_gradient, niterations=self.niterations)[0]

    def apply(self, x):
        return sigmoid(self.theta.T @ x)

import numpy as np
from Classification import proposed
from save_load import load


def fit_func_70(x):
    x_train = load('x_train_70')
    x_test = load('x_test_70')
    y_train = load('y_train_70')
    y_test = load('y_test_70')

    epochs = int(x[0])
    learning_rate = x[2]
    batch_size = int(x[1])

    pred, met, history = proposed(x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate)

    fit = 1 / met[0]

    return fit


def fit_func_80(x):
    x_train = load('x_train_80')
    x_test = load('x_test_80')
    y_train = load('y_train_80')
    y_test = load('y_test_80')

    epochs = int(x[0])
    learning_rate = x[2]
    batch_size = int(x[1])

    pred, met, history = proposed(x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate)

    fit = 1 / met[0]

    return fit

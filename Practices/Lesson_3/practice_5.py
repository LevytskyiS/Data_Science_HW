import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def h(X, W):
    return np.dot(X, W)


def loss_function(X, Y, W):
    m = X.shape[0]
    return np.square(h(X, W) - Y).sum() / (2 * m)


def grad_step(W, grad_w, learning_rate=0.001):
    W = W - learning_rate * grad_w
    return W


def grad(X, Y, W):
    m = X.shape[0]
    np.dot(X.T, (h(X, W) - Y)) / m
    return np.dot(X.T, (h(X, W) - Y)) / m


def grad_descent(X, Y, W, num_iter=10000, learning_rate=0.001, epsilon=0.0000001):
    loss = loss_function(X, Y, W)
    loss_history = [loss]
    for i in range(num_iter):
        best = None
        grad_w = grad(X, Y, W)
        W = grad_step(W, grad_w, learning_rate=learning_rate)
        loss = loss_function(X, Y, W)
        if abs(loss - loss_history[-1]) < epsilon:
            loss_history.append(loss)
            best = grad_w
            break
        loss_history.append(loss)
    return W, best, loss_history


df = pd.read_csv("Housing.csv")


def normalize(value):
    return (value - value.mean()) / value.std()


normalized_table = pd.DataFrame()

normalized_table["price"] = normalize(df["price"])
normalized_table["area"] = normalize(df["area"])
normalized_table["bedrooms"] = normalize(df["bedrooms"])
normalized_table["bathrooms"] = normalize(df["bathrooms"])


Y = normalized_table["price"].values.reshape(-1, 1)
X = normalized_table[["area", "bathrooms", "bedrooms"]].values
X = np.hstack((np.ones((X.shape[0], 1)), X))
N = X.shape[1]
W = np.linspace(0, 0, N).reshape((N, 1))
W, best, loss_history = grad_descent(X, Y, W, 10000, learning_rate=0.001)
loss = loss_history[-1]
print(f"Best values: {best}")
print(f"Loss func: {loss}")

weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
analytical = loss_function(X, Y, weights)

print(f"Best values: {weights}")
print(f"Analytical value of loss func: {analytical} and value of loss function {loss} ")

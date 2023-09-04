import random

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

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

print(X)

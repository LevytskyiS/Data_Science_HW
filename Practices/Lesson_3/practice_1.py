import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

housing = "Housing.csv"
df = pd.read_csv(housing)

# df.plot(x="area", y="price", kind="scatter", figsize=(12, 8))
# plt.show()

# sns.set_style("darkgrid")
# sns.pairplot(
#     df, vars=["price", "area", "bedrooms", "bathrooms", "stories"], hue="basement"
# )
# plt.show()
df_corr = df[["area", "price"]].corr()


# Linear regression
def h(w_0, w_1, x):
    return w_0 + w_1 * x


# Compute Loss function
def loss_function(w_0, w_1, df):
    n = df.area.shape[0]
    cost = 0
    for x, y in zip(df.area, df.price):
        cost = cost + (h(w_0, w_1, x) - y) ** 2
    return cost / (2 * n)


# w_0 = 0
# w_1 = np.linspace(-6000, 8000, 500)

# plt.plot(w_1, [loss_function(w_0, w, df) for w in w_1])
# plt.show()

grid_w_0 = np.arange(-2000, 2000, 10)
grid_w_1 = np.arange(-10000, 10000, 10)
# grid_w_2 = np.arange(-10000, 10000, 10)

w_0, w_1 = np.meshgrid(grid_w_0, grid_w_1)

print(w_0)
print(w_1)


# z = loss_function(w_0, w_1, df)

# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(w_0, w_1, z)
# plt.show()

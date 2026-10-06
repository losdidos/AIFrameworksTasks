import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv("ex1data2.txt", header=None)




X = data.iloc[:, :2].to_numpy(dtype=float)
y = data.iloc[:, 2].to_numpy(dtype=float)



model = LinearRegression().fit(X, y)



print("Scikit-learn coefficients:", model.coef_)


# Plot made with ai
size_grid, bedrooms_grid = np.meshgrid(
	np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
	np.linspace(X[:, 1].min(), X[:, 1].max(), 20),
)
price_grid = model.predict(
	np.c_[size_grid.ravel(), bedrooms_grid.ravel()]
).reshape(size_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y.ravel(), color="blue", label="Data points")
ax.plot_surface(size_grid, bedrooms_grid, price_grid, alpha=0.4, color="orange")
ax.set_xlabel("House size (sq ft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")
ax.set_title("House Prices and Linear Regression")
plt.show()
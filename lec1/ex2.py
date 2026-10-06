from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


data = np.loadtxt("ex1data1.txt", delimiter=",")
X = data[:, 0].reshape(-1, 1)  
y = data[:, 1]                 

model = LinearRegression().fit(X, y)

print("Theta 0 (intercept):", model.intercept_)
print("Theta 1 (slope):", model.coef_[0])





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
regressor_first = LinearRegression()
regressor_first.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)

regressor_second = LinearRegression()
regressor_second.fit(X_poly, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, regressor_second.predict(X_poly), color='blue')
plt.show()
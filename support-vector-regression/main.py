### Support Vector Regression ###

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

# Feature Scaling - Some models automatically take care of feature scaling but SVR doesn't. So we need to do it manually before feeding it to SVR model
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Applying SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Prediction
scaled_prediction = regressor.predict(sc_x.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(scaled_prediction.reshape(-1, 1))

print(y_pred)

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()
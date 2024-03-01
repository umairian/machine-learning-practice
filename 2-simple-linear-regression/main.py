## Simple Linear Regression ##

### Step 0: Importing Libraries ###
import pandas as pd
import matplotlib.pyplot as plt

### Step 1: Importing dataset ###
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
print(X)
Y=dataset.iloc[:, -1].values

### Step 2: Split dataset in train test parts ###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 1/3, random_state=0)

### Step 3: Fitting simple linear regression model to the training set ###
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

### Step 4: Predicting the outcome on test set ###
predictions = regressor.predict(x_test)

# Plotting the data on the graph
plt.scatter(x_test, y_test)
plt.plot(x_test, predictions, color="green")
plt.show()

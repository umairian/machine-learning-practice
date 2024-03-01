#### DATA PRE-PROCESSING ####

### Step 0: Importing libraries ###

# NumPy library is used for numerical computating. It provides support for multi-dimesional arrays and matrices, along with mathematical functions to operate on these arrays efficiently
import numpy as np
# Pandas library is used for data manipulation and analysis as it provides structures like DataFrame and Series that handles structured data efficiently. Commonly used in ML and data analysis
import pandas as pd

### Step 1: Importing the Dataset ###
dataset = pd.read_csv('Data.csv')
# The iloc function in pandas is used to select data by position
# .values converts the selected data into a NumPy array.
# Difference between Lists and Arrays: Python lists can contain elements of different data types, while arrays in Python (specifically NumPy arrays) are homogeneous and contain elements of the same data type.
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

### Step 2: Handling Missing Data ###
# In English, the word "impute" means to attribute or ascribe something to someone or something
# In the context of data processing, "impute" refers to the process of replacing missing or null values in a dataset with substituted values. This is done to ensure that the dataset is complete and can be used for analysis or modeling without errors caused by missing data.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# imputer.fit() is used to fit the imputer instance on the data provided. This means it computes the mean, median, or most frequent value of the data along with any other parameters required for imputation.
imputer = imputer.fit(X[:, 1:3])
# imputer.transform() is used to apply the imputation. It replaces missing values in the data with the computed values during the fitting process. This step actually transforms the data by replacing missing values with the imputed values.
X[:, 1:3] = imputer.transform(X[:, 1:3])

### Step 3: Encoding Categorical Data ###
# Categorical data is data that represents categories or labels, rather than numerical values or a finite set on values. For example in a dataset, Country column has three or four countries.
# To fed the dataset into a model, we need to encode the categorical data into numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# If a column has more than two categorical values, we need to split it into multiple columns of boolean values
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

### Step 4: Splitting into Training and Testing sets ###
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=0)

### Step 5: Feature Scaling ###
# Feature scaling is used to standardize the range of independent variables or features of data. It ensures that all features have the same scale, which can improve the performance of machine learning algorithms that are sensitive to the scale of the input features (most models work on the Eucledian distance, so if a column has large scale compared to others, it will dominate other data in the computation). Common methods of feature scaling include normalization and standardization.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
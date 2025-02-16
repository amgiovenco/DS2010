# Homework 3 DS 2010
# Alessandra Giovenco

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# Section 2.4 Exercise 7 - KNN
data = np.array ([
    [0, 3, 0, 'Red'],
    [2, 2, 0, 'Red'],
    [0, 0, 3, 'Red'],
    [0, 1, 1, 'Green'],
    [-1, 0, 1, 'Green'],
    [1, 1, 1, 'Red']
])

dfKNN = pd.DataFrame(data, columns = ['x1', 'x2', 'x3', 'y'])
dfKNN[['x1', 'x2', 'x3']] = dfKNN[['x1', 'x2', 'x3']].astype(float)

test_point = np.array([0, 0, 0])
dfKNN['Distance'] = np.linalg.norm(dfKNN[['x1', 'x2', 'x3']].values - test_point, axis=1)

dfKNNSorted =  dfKNN.sort_values(by = ['Distance'])

# K = 1 Prediction
k1Prediction = dfKNNSorted.iloc[0]['y']

# K = 3 Prediction
k3Prediction = dfKNNSorted.iloc[:3]['y'].mode()[0]

print("*Section 2.4 Exercise 7 - KNN")
print("KNN Prediction for K = 1: ", k1Prediction)
print("KNN Prediction for K = 3: ", k3Prediction)

# Section 3.7 Exercise 4 - RSS
def genData(n = 100):
    X = np.random.uniform(-5, 5, size = n)
    Y = 3 + 2 * X + np.random.normal(scale = 2, size = n)
    return X, Y

X, Y = genData()
XCubic = np.vstack([X, X**2, X**3]).T

linModel = LinearRegression().fit(X.reshape(-1, 1), Y)
cubeModel = LinearRegression().fit(XCubic, Y)

linRSS = mean_squared_error(Y, linModel.predict(X.reshape(-1, 1))) * len(Y)
cubeRSS = mean_squared_error(Y, cubeModel.predict(XCubic)) * len(Y)

print("\n*Section 3.7 Exercise 4 - RSS")
print("RSS for Linear Model: ", linRSS)
print("RSS for Cubic Model: ", cubeRSS)

# Section 3.7 Exercise 8 - Simple Linear Regression


print("\n*Section 3.7 Exercise 8 - Simple Linear Regression")


# Section 4.8 Exercise 13 - Logistic Regression & Classification


print("\n*Section 4.8 Exercise 13 - Logistic Regression & Classification")
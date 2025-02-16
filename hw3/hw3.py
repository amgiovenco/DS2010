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
auto = sm.datasets.get_rdataset("Auto", "ISLR").data
model = sm.OLS(auto['mpg'], sm.add_constant(auto['horsepower'])).fit()

# Regression plot
plt.figure(figsize = (8, 6))
sns.regplot(x = auto['horsepower'], y = auto['mpg'], scatter_kws = {'alpha': 0.5})
plt.title("MPG v. Horsepower")
print("\n*Section 3.7 Exercise 8 - Simple Linear Regression")
print(model.summary())
plt.show()

# Section 4.8 Exercise 13 - Logistic Regression & Classification
print("\n*Section 4.8 Exercise 13 - Logistic Regression & Classification")

weekly = sm.datasets.get_rdataset("Weekly", "ISLR").data
X = weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = weekly['Direction'].apply(lambda x: 1 if x == 'Up' else 0)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)

logRegression = LogisticRegression().fit(XTrain, yTrain)
yPrediciton = logRegression.predict(XTest)
print("Logistic Regression Accuracy: ", accuracy_score(yTest, yPrediciton))

# Confusion Matrix 
print("Confusion Matrix:\n", confusion_matrix(yTest, yPrediciton))

# LDA
lda = LinearDiscriminantAnalysis().fit(XTrain, yTrain)
yPredicitonLDA = lda.predict(XTest)
print("LDA Accuracy: ", accuracy_score(yTest, yPredicitonLDA))

# QDA
qda = QuadraticDiscriminantAnalysis().fit(XTrain, yTrain)
yPredicitonQDA = qda.predict(XTest)
print("QDA Accuracy: ", accuracy_score(yTest, yPredicitonQDA))

# KNN where K = 1
knn = KNeighborsClassifier(n_neighbors = 1).fit(XTrain, yTrain)
yPredicitonKNN = knn.predict(XTest)
print("KNN Accuracy where K = 1: ", accuracy_score(yTest, yPredicitonKNN))

# Naive Bayes
nb = GaussianNB().fit(XTrain, yTrain)
yPredicitonNB = nb.predict(XTest)
print("Naive Bayes Accuracy: ", accuracy_score(yTest, yPredicitonNB))
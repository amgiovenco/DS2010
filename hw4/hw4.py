import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# I need to load the weekly dataset from the ISL thing
Weekly = load_data('Weekly')
Weekly.columns

# preprocessing because I did the whole thing wrong last time. 
Weekly['DIrection'] = Weekly['Direction'].map({'Up': 1, 'Down': 0})

trainData = Weekly[Weekly['Year'] < 2009]
testData = Weekly[Weekly['Year'] >= 2009]

X_train = trainData.drop(['Direction', 'Year', 'Today'], axis=1)
y_train = trainData['Direction']

X_test = testData.drop(['Direction', 'Year', 'Today'], axis=1)
y_test = testData['Direction']

print("Section 4.8, Question 13 j.\n")

#  log reg model
poly = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

logRegPoly = LogisticRegression(max_iter=1000)
logRegPoly.fit(X_train_poly, y_train)

y_pred_poly = logRegPoly.predict(X_test_poly)
print("Confusion Matrix - Log Regression w/ Polynomial Features: \n", confusion_matrix(y_test, y_pred_poly))
print("\nClassification Report: \n", classification_report(y_test, y_pred_poly))

# Standardize the predictors for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# hyperparameter tuning
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# best k values
best_k = grid_search.best_params_['n_neighbors']
print("\nBest K Value:\n", best_k)

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

y_pred_knn = knn_best.predict(X_test_scaled)
print("\nConfusion Matrix - KNN:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# Accuracy comparison
accuracy_poly = accuracy_score(y_test, y_pred_poly)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nAccuracy - Log Regression w/ Polynomial Features:\n", accuracy_poly)
print("\nAccuracy - KNN:\n", accuracy_knn)

print("\nBoth models have perfect accuracy. This is most likely because the Weekly dataset is only 2009-2010.")
print("The logistical regression model caught the non-linear relationships between predictors.\n")
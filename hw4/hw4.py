import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import statsmodels.api as sm
import patsy as pt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from ISLP import load_data
from IPython.display import display, HTML

print("Section 4.8 Question 13 j:\n")
# I need to load the weekly dataset from ISL package
Weekly = load_data('Weekly')
Weekly.columns
Weekly['Direction'] = Weekly['Direction'].map({'Up': 1, 'Down': 0})


def confusionTable(confusion_mtx):
    """Renders a nice confusion table with labels"""
    confusion_df = pd.DataFrame({'y_pred=0': np.append(confusion_mtx[:, 0], confusion_mtx.sum(axis=0)[0]),
                                 'y_pred=1': np.append(confusion_mtx[:, 1], confusion_mtx.sum(axis=0)[1]),
                                 'Total': np.append(confusion_mtx.sum(axis=1), ''),
                                 '': ['y=0', 'y=1', 'Total']}).set_index('')
    return confusion_df

train_idx = Weekly.index[Weekly['Year'] < 2009]
WeeklyTrain = Weekly.iloc[train_idx]
WeeklyTest  = Weekly.drop(train_idx)

predictors  = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Year']
X_train = np.array(WeeklyTrain[predictors])
X_test  = np.array(WeeklyTest[predictors])
y_train = np.array(WeeklyTrain['Direction'])
y_test  = np.array(WeeklyTest['Direction'])

# Logistic Regression
#model_logit = sm.Logit(y_train, X_train).fit() <--- this technique didn't converge
logit       = LogisticRegression()
model_logit = logit.fit(X_train, y_train)
# LDA
lda         = LinearDiscriminantAnalysis()
model_lda   = lda.fit(X_train, y_train)
# QDA
qda         = QuadraticDiscriminantAnalysis()
model_qda   = qda.fit(X_train, y_train)
# KNN_1
K = 1
model_knn_1 = KNeighborsClassifier(n_neighbors=K).fit(preprocessing.scale(X_train), y_train)
# KNN_3
K = 3
model_knn_3 = KNeighborsClassifier(n_neighbors=K).fit(preprocessing.scale(X_train), y_train)
# KNN_10
K = 10
model_knn_10 = KNeighborsClassifier(n_neighbors=K).fit(preprocessing.scale(X_train), y_train)

models = {'logit': model_logit, 
          'lda': model_lda, 
          'qda': model_qda,
          'knn_1': model_knn_1,
         'knn_3': model_knn_3,
         'knn_10': model_knn_10}
scaled = ['knn_1', 'knn_3', 'knn_10']

# Predict
for k in models:
    if k in scaled:
        y_pred = models[k].predict(preprocessing.scale(X_test))
    else:
        y_pred = models[k].predict(X_test)
    # Confusion table
    display(HTML('<h3>{}</h3>'.format(k)))
    confusion_mtx = confusion_matrix(y_test, y_pred)
    display(confusionTable(confusion_mtx))

print("\n\nSection 8.4 Question 8 a & b:\n")
# Split the data set into a training set and a test set.
Carseats = load_data('Carseats')
Carseats.columns

# Check for missing values
assert Carseats.isnull().sum().sum() == 0

# Drop unused index if it exists
if 'Unnamed: 0' in Carseats.columns:
    Carseats = Carseats.drop('Unnamed: 0', axis=1)

# Create index for training set
np.random.seed(1)
train = np.random.random(len(Carseats)) > 0.5

display(Carseats.head())

preds = Carseats.columns.drop(['Sales'])
f = 'Sales ~ 0 +' + ' + '.join(preds)
y, X = pt.dmatrices(f, Carseats)
y = y.flatten()

# Fit Sklearn's tree regressor
clf = tree.DecisionTreeRegressor(max_depth=5).fit(X[train], y[train])

# Measure test set MSE
y_hat = clf.predict(X[~train])
mse = metrics.mean_squared_error(y[~train], y_hat)

# Get proportion of correct classifications on test set
print('Test MSE: {}'.format(np.around(mse, 3)))
print('Test RMSE: {}'.format(np.around(np.sqrt(mse), 3)))
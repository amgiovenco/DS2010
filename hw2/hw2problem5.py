# Homework 2 DS 2010
# Alessandra Giovenco

import numpy as np

# Matrix X =
X = np.array([
    [-8, -3, 84, 6],
    [-21, 9, 0, 9],
    [5, 0, 102, -3],
    [-2, -6, -4, 2],
    [5, 8, 35, 6],
    [-15, 1, 54, 3],
    [-8, 7, 14, 5],
    [3, 1, 50, 14],
    [17, 1, 92, 4]
])

# 5a)
meanX = np.mean(X, axis = 0)

# 5b)
nSamples, nFeatures = X.shape
covMatrixDims = (nFeatures, nFeatures)

# 5c)
# Center Matrix.
XCentered = X - meanX

# Compute sample covariance matrix
sampleCovMatrix = (XCentered.T @ XCentered) / (nSamples - 1)

# Results! 
print(f"5a)\nMean of X:\n{meanX}")
print(f"\n5b)\nCovariance Matrix Dimensions: {covMatrixDims}")
print(f"\n5c)\nSample Covariance Matrix:\n{sampleCovMatrix}")
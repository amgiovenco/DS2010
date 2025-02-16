# Homework 2 DS 2010
# Alessandra Giovenco

import numpy as np

# 6)
xVals6 = [4, 6, 8, 10]
pVals6 = [0.2, 0.1, 0.3, 0.4]

# 6a)
pXeq6 = pVals6[xVals6.index(6)]
pXlessEq7 = sum(p for x, p in zip(xVals6, pVals6) if x <= 7)
pXgt7 = sum(p for x, p in zip(xVals6, pVals6) if x > 7)

# 6b)
EX6 = sum(x * p for x, p in zip(xVals6, pVals6))
E7plus10X = 7 + 10 * EX6

# 6c)
e1overX = sum((1 / x) * p for x, p in zip(xVals6, pVals6))

# Results!
print(f"Problem 6:")
print(f"6a)\nP(X = 6) = {pXeq6}")
print(f"P(X <= 7) = {pXlessEq7}")
print(f"P(X > 7) = {pXgt7}")
print(f"\n6b)\nE(X) = {EX6}")
print(f"E(7 + 10X) = {E7plus10X}")
print(f"\n6c)\nE(1/X) = {e1overX}")

# 7)
xValues7 = [0, 1, 2, 3]
pCoeffs7 = [1, 3, 2, 4]

# 7a)
totalCoeff = sum(pCoeffs7)
p = 1 / totalCoeff
pValues7 = [p * pCoeff for pCoeff in pCoeffs7]

# 7b)
EX7 = sum(x * p for x, p in zip(xValues7, pValues7))

# 7c)
eX2_7 = sum((x**2) * p for x, p in zip(xValues7, pValues7))
stdX7 = np.sqrt(eX2_7 - EX7**2)

# 7d)
E100minus10X = 100 - 10 * EX7

# 7e)
std100minus10X = 10 * stdX7

# Results!
print(f"\n\nProblem 7:")
print(f"7a)\nProbability Values: {pValues7}")
print(f"\n7b)\nE(X) = {EX7}")
print(f"\n7c)\nE(X^2) = {eX2_7}\nStandard Deviation of X = {stdX7}")
print(f"\n7d)\nE(100 - 10X) = {E100minus10X}")
print(f"\n7e)\nStandard Deviation of 100 - 10X = {std100minus10X}")

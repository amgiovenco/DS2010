# Homework 2 DS 2010
# Alessandra Giovenco

import math

# Given values
EX = 1
EY = -3
EX2 = 5
EY2 = 11
EXY = -1

# 3a)
EXplusY = EX + EY

# 3b)
CovXY = EXY - (EX * EY)

# 3c)
varX = EX2 - EX**2
varY = EY2 - EY**2
StdX = math.sqrt(varX)
StdY = math.sqrt(varY)
CorrXY = CovXY / (StdX * StdY)

# Results!
print(f"3a)\nE[X + Y]: {EXplusY}")
print(f"\n3b)\nCov(X, Y): {CovXY}")
print(f"\n3c)\nCorr(X, Y): {CorrXY}")
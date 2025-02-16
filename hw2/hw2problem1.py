# Homework 2 DS 2010
# Alessandra Giovenco

import pandas as pd
import numpy as np

# Question 1 
data = {
    "X": [-2, 0, -2, 0, 2, 4, 0],
    "Y": [-2, 0, 0, 2, -2, -4, 6],
    "P_XY": [1/20, 2/20, 4/20, 7/20, 1/20, 1/20, 2/20]
}
df = pd.DataFrame(data)

# 1a)
PXneg2 = df[df["X"] == -2]["P_XY"].sum()
PYneg2 = df[df["Y"] == -2]["P_XY"].sum()
PXneg2GivenYneg2 = df[(df["X"] == -2) & (df["Y"] == -2)]["P_XY"].iloc[0] / PYneg2

# 1b)
PX = df.groupby("X")["P_XY"].sum()
PY = df.groupby("Y")["P_XY"].sum()

# 1c)
EX = sum(x * p for x, p in zip(PX.index, PX))
EY = sum(y * p for y, p in zip(PY.index, PY))
EXY = sum(row["X"] * row["Y"] * row["P_XY"] for _, row in df.iterrows())
covarianceXY = EXY - (EX * EY)

# 1d)
independence = all(
    np.isclose(row["P_XY"], PX[row["X"]] * PY[row["Y"]])
    for _, row in df.iterrows()
)

# Results!
print(f"1a)\nP(X = -2): {PXneg2}")
print(f"\nP(x = -2 | Y = -2): {PXneg2GivenYneg2}")
print(f"\n1b)\nMarginal PMF of X: {PX}")
print(f"\nMarginal PMF of Y: {PY}")
print(f"\n1c)\nE[X]: {EX}\nE[Y]: {EY}\nE[XY]: {EXY}\nCovarience of X and Y: {covarianceXY}")
print(f"\n1d)\nAre X and Y independent? {'Yes' if independence else 'No'}")
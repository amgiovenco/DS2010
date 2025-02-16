# Homework 2 DS 2010
# Alessandra Giovenco

import numpy as np
from scipy.integrate import quad

# 2a)
def cdf(x):
    """Cumulative distribution function of X."""
    if x <= 2:
        return 0
    else:
        return 1 - (16 / x**4)
    
def pdf(x):
    """Probability density function of X."""
    if x <= 2:
        return 0
    else:
        return (64 / x**5)
    
# 2b)
# i.
prob_cdf = cdf(8) - cdf(4)

# ii.
prob_pdf, _ = quad(pdf, 4, 8)

# 2c)
expValues, _ = quad(lambda x: x * pdf(x), 2, np.inf)

# Results!
print(f"2a)\nThe PDF of X is f(x) = 64 / x^5 for x > 2")
print(f"\n2b)\nProbability that 4 < X < 8:")
print(f"\n i. CDF: {prob_cdf}")
print(f"\n ii. PDF: {prob_pdf}")
print(f"\n2c)\nThe expected wait time of (E[X]): {expValues}")


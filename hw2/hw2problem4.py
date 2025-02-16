# Homework 2 DS 2010
# Alessandra Giovenco

# 4) 
# A new type of electronic flash bulb for cameras lasts an average of 5000 hours with a
# standard deviation of 500 hours. A quality control engineer selects a random sample of these
# flash bulbs with sample size 100. Approximate the probability that the sample mean lifetime
# will be less than 4928 hours. (hint: use the central limit theorem)

import numpy as np
from scipy.stats import norm

# Given values
mu = 5000
sigma = 500
n = 100
sampleMu = 4928

# 4)
# The sample mean lifetime is normally distributed with mu and sigma / sqrt(n)
sampleSigma = sigma / np.sqrt(n)
z = (sampleMu - mu) / sampleSigma
prob = norm.cdf(z)

# Results!
print(f"4)\nProbability that the sample mean lifetime\nwill be less than 4928 hours: {prob}")

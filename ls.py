import numpy as np

maxEpsilon = 1
minEpsilon = 0.01
decayRate = 0.00001

for x in range(100000):
    epsilon = minEpsilon + (maxEpsilon - minEpsilon) * np.exp(-decayRate * x)
    print(epsilon)

import data
import numpy as np


def func(x):
    return data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(
        -data.decayRate * x
    )


i = 0

while True:
    y = func(i)
    i += 1
    if y <= data.minEpsilon:
        print(i)
        break

import numpy as np

X = np.array([
    [-2.0, 1.0, 3.0],
    [4.0, -5.0, 2.0]
])

output = np.maximum(X,0)

print(output)
import numpy as np
from src.layers import Dense

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.5],
    [0.5, 1.5, 2.0],
    [3.0, 1.0, 2.0],
])

layer = Dense(3, 4)

output = layer.forward(X)

print("X:", X.shape)
print("W:", layer.W.shape)
print("b:", layer.b.shape)
print("output:", output.shape)
print(output)
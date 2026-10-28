import numpy as np

from src.layers import Dense, ReLU


X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.5],
    [0.5, 1.5, 2.0],
    [3.0, 1.0, 2.0],
])

dense1 = Dense(3, 4)
relu1 = ReLU()

Z1 = dense1.forward(X)
A1 = relu1.forward(Z1)

print("X:", X.shape)
print("Z1:", Z1.shape)
print("A1:", A1.shape)

print("\nZ1:")
print(Z1)

print("\nA1:")
print(A1)

dense2 = Dense(4, 2)

Z2 = dense2.forward(A1)

print("\nZ2:", Z2.shape)
print(Z2)
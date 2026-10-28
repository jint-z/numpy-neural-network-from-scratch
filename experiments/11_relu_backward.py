import numpy as np

from src.layers import ReLU


X = np.array([
    [-2.0, 1.0, 3.0],
    [4.0, -5.0, 6.0],
])

dY = np.array([
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0],
])


relu = ReLU()


# Forward
Y = relu.forward(X)

print("X:")
print(X)

print("\nY:")
print(Y)


# Backward
dX = relu.backward(dY)

print("\ndY:")
print(dY)

print("\ndX:")
print(dX)

print("\ndX shape:", dX.shape)
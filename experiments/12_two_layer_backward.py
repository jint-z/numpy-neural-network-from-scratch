import numpy as np

from src.layers import Dense, ReLU


np.random.seed(42)

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
])

y_true = np.array([
    [3.0],
    [3.0],
    [7.0],
    [7.0],
])


# ====================
# Network
# ====================

dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 1)


# ====================
# Forward
# ====================

z1 = dense1.forward(X)
a1 = relu.forward(z1)
y_pred = dense2.forward(a1)

loss = np.mean((y_pred - y_true)**2)


# ====================
# Backward
# ====================

N = X.shape[0]

d_y_pred = 2*(y_pred - y_true) / N

d_a1 = dense2.backward(d_y_pred)
d_z1 = relu.backward(d_a1)
dX = dense1.backward(d_z1)


# ====================
# Output
# ====================

print("y_pred:")
print(y_pred)

print("\nloss:")
print(loss)

print("\ndense2 dW shape:", dense2.dW.shape)
print("dense2 db shape:", dense2.db.shape)

print("\ndense1 dW shape:", dense1.dW.shape)
print("dense1 db shape:", dense1.db.shape)

print("\ndX shape:", dX.shape)
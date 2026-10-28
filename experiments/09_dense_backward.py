import numpy as np

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
])

b = np.array([0.1, 0.2, 0.3])

# forward
Y = X @ W + b

print("Y:")
print(Y)
print("Y shape:", Y.shape)


# 假设这是后面网络传回来的梯度
dY = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])


# TODO 1
# 计算 W 的梯度
dW = X.T@dY


# TODO 2
# 计算 b 的梯度
db = np.sum(dY,axis=0)

# TODO 3
# 计算 X 的梯度
dX = dY@W.T


print("\ndW:")
print(dW)
print("dW shape:", dW.shape)

print("\ndb:")
print(db)
print("db shape:", db.shape)

print("\ndX:")
print(dX)
print("dX shape:", dX.shape)
import numpy as np

from src.layers import Dense


X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
])

b = np.array([0.1, 0.2, 0.3])

dY = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])


# 创建 Dense
dense = Dense(2, 3)

# 为了方便验证，覆盖随机初始化参数
dense.W = W.copy()
dense.b = b.copy()


# ====================
# Forward
# ====================

Y = dense.forward(X)

print("Y:")
print(Y)


# ====================
# Backward
# ====================

dX = dense.backward(dY)

print("\ndW:")
print(dense.dW)

print("\ndb:")
print(dense.db)

print("\ndX:")
print(dX)
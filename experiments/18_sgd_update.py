import numpy as np
from src.layers import Dense

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

dense = Dense(2, 3)

Y = dense.forward(X)
#保存旧参数
old_W = dense.W.copy()
old_b = dense.b.copy()

#构造dY
dY = np.ones_like(Y)

dX = dense.backward(dY)

#设置学习率
learning_rate = 0.1

#更新参数
dense.W = dense.W - learning_rate * dense.dW
dense.b = dense.b -learning_rate * dense.db 

#打印
print("old_W:",old_W,"\nnew_W:",dense.W)
print("old_b:",old_b,"\nnew_b:",dense.b)
print("dW:",dense.dW,"\ndb:",dense.db)

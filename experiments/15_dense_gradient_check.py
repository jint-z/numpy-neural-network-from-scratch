import numpy as np
from src.layers import Dense

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)

    for i in np.ndindex(x.shape):
        original = x[i]

        # Step 1
        # x[i] 设置成 original + eps
        # 计算 loss_plus
        x[i] = original + eps
        loss_plus = f(x)

        # Step 2
        # x[i] 设置成 original - eps
        # 计算 loss_minus
        x[i] = original - eps
        loss_minus = f(x)

        # Step 3
        # 使用中心差分公式计算 grad[i]
        grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Step 4
        # 恢复 x[i]
        x[i]=original

    return grad
def f_w(W):
    dense.W = W
    Y = dense.forward(X)
    loss = np.sum(Y)
    return loss
def f_b(b):
    dense.b = b
    Y = dense.forward(X)
    loss = np.sum(Y)
    return loss
def f_x(x):
    Y = dense.forward(x)
    loss = np.sum(Y)
    return loss



X = np.array([[1.,2.],[3.,4.]])
dense = Dense(2,3)
Y = dense.forward(X)
loss = np.sum(Y)
dY = np.ones_like(Y)
dX = dense.backward(dY)
#检查dW
analytical_dW = dense.dW
numerical_dW = numerical_gradient(f_w,dense.W)
print("analytical_dW: ",analytical_dW)
print("numerical_dW: ",numerical_dW)
print("difference:",abs(analytical_dW-numerical_dW))

#检查db
analytical_db = dense.db
numerical_db = numerical_gradient(f_b,dense.b)
print("analytical_db: ",analytical_db)
print("numerical_db: ",numerical_db)
print("difference:",abs(analytical_db-numerical_db))

#检查dX
analytical_dX = dX
numerical_dX = numerical_gradient(f_x,X)
print("analytical_dX: ",analytical_dX)
print("numerical_dX: ",numerical_dX)
print("difference:",abs(analytical_dX-numerical_dX))

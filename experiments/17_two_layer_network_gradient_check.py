import numpy as np
from src.layers import Dense, ReLU

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

def forward_loss():
    Z1 = dense1.forward(X)
    A1 = relu.forward(Z1)
    y_pred = dense2.forward(A1)
    loss = np.mean((y_pred-y_true)**2)
    return loss
def f_W1(W):
    dense1.W = W
    return forward_loss()
def f_W2(W):
    dense2.W = W
    return forward_loss()
def f_b1(b):
    dense1.b = b
    return forward_loss()
def f_b2(b):
    dense2.b = b
    return forward_loss()
X = np.array([
    [1.0, -2.0],
    [-3.0, 4.0],
    [2.0, 1.0]
])

y_true = np.array([
    [1.0],
    [-1.0],
    [0.5]
])

dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 1)

dense1.W = np.array([
    [0.2, -0.3, 0.4],
    [0.5,  0.1, -0.2]
])

dense1.b = np.array([0.1, -0.2, 0.3])

dense2.W = np.array([
    [0.6],
    [-0.4],
    [0.2]
])

dense2.b = np.array([0.1])

Z1 = dense1.forward(X)

A1 = relu.forward(Z1)

y_pred = dense2.forward(A1)

errors = y_pred-y_true

loss =np.mean(errors**2)

d_y_pred = 2*(y_pred-y_true)/y_pred.size

dA1 = dense2.backward(d_y_pred)

dZ1 = relu.backward(dA1)

dX = dense1.backward(dZ1)

analytical_dW1 = dense1.dW.copy()
analytical_dW2 = dense2.dW.copy()
analytical_db1 = dense1.db.copy()
analytical_db2 = dense2.db.copy()

#检查
numerical_dW1 = numerical_gradient(f_W1,dense1.W)
numerical_dW2 = numerical_gradient(f_W2,dense2.W)
numerical_db1 = numerical_gradient(f_b1,dense1.b)
numerical_db2 = numerical_gradient(f_b2,dense2.b)

print("dW1_difference:",np.abs(analytical_dW1-numerical_dW1))
print("db1_difference:",np.abs(analytical_db1-numerical_db1))
print("dW2_difference:",np.abs(analytical_dW2-numerical_dW2))
print("db2_difference:",np.abs(analytical_db2-numerical_db2))



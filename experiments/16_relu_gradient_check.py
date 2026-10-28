import numpy as np
from src.layers import ReLU


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


np.random.seed(42)

X = np.array([
    [-2.0, 1.0, 3.0],
    [4.0, -5.0, 6.0]
])
dY = np.array([
    [1.0, -2.0, 3.0],
    [-4.0, 5.0, -6.0]
]) 

relu = ReLU()

# 1. forward
Y = relu.forward(X)

# 2. analytical gradient
analytical_dX = relu.backward(dY)


# 3. 定义用于 numerical gradient 的 scalar function
def f(x):
    Y = relu.forward(x)
    loss = np.sum(dY*Y)
    return loss


# 4. numerical gradient
numerical_dX = numerical_gradient(f,X)


# 5. 比较
print("numerical_dX:",numerical_dX)
print("analytical_dX:",analytical_dX)
print("difference:",np.abs(numerical_dX-analytical_dX))

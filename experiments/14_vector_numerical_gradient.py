import numpy as np
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)

    for i in range(x.size):
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

x = np.array([1.0, 2.0, 3.0])
def f(x):
    return np.sum(x ** 2)
print(numerical_gradient(f,x,))
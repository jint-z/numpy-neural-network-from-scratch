import numpy as np

y_true = np.array([
    [3.0],
    [5.0],
    [7.0],
])

y_pred = np.array([
    [2.5],
    [5.5],
    [8.0],
])

errors = y_pred - y_true

N = y_true.shape[0]

# TODO:
# 根据刚刚推导的公式计算
# dL / dy_pred
gradient = 2*errors/N

print("gradient:")
print(gradient)
print("shape:", gradient.shape)

learning_rate = 0.1

old_loss = np.mean((y_pred - y_true) ** 2)

learning_rate = 0.1

# TODO
new_y_pred = y_pred-learning_rate*gradient

new_loss = np.mean((new_y_pred - y_true) ** 2)

print("\nold predictions:")
print(y_pred)

print("\nnew predictions:")
print(new_y_pred)

print("\nold loss:", old_loss)
print("new loss:", new_loss)
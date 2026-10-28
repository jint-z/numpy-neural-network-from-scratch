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

# TODO 1:
# 计算每个样本 prediction - target
errors = (y_pred-y_true)

print("errors:")
print(errors)
print("shape:", errors.shape)

# TODO 2:
# 对每个误差平方
squared_errors = errors*errors

print("\nsquared errors:")
print(squared_errors)

# TODO 3:
# 对所有平方误差求平均
loss = np.average(squared_errors)

print("\nMSE loss:")
print(loss)
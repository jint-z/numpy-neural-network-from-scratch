import numpy as np

from src.layers import Dense
from src.losses import MSELoss
from src.optimizers import SGD


# 【目标】观察方向相反的单样本梯度如何影响批次更新。
# 【输入/输出】X (2,1)、y (2,1)，单输出 Dense；打印 loss、dW、db 和更新后参数。
# 【步骤 1】固定 W=0、b=0，分别计算并打印两个样本的损失和梯度。
# 【步骤 2】不更新参数，计算两个样本一起构成的批次损失和梯度。
# 【步骤 3】用 SGD 对批次梯度更新一次，比较更新前后的 W、b 和批次损失。
# 【验收】两个单样本梯度非零且方向相反；批次梯度为零，参数和损失不变。
# 先预测，再运行；所有标签均为实验构造数据。

dense = Dense(1, 1)
dense.W = np.array([[0.0]])
dense.b = np.array([0.0])
loss_fn = MSELoss()
optimizer = SGD(learning_rate=0.1)
X = np.array([[1.0], [1.0]])
y = np.array([[1.0], [-1.0]])

for i in range(2):
    prediction = dense.forward(X[i:i + 1])
    loss = loss_fn.forward(prediction, y[i:i + 1])
    dense.backward(loss_fn.backward(prediction, y[i:i + 1]))
    print("sample", i + 1, "loss:", loss, "dW:", dense.dW.copy(), "db:", dense.db.copy())

prediction = dense.forward(X)
batch_loss = loss_fn.forward(prediction, y)
dense.backward(loss_fn.backward(prediction, y))
print("batch loss:", batch_loss, "dW:", dense.dW, "db:", dense.db)
optimizer.step(dense)
new_prediction = dense.forward(X)
print("after update W:", dense.W, "b:", dense.b)
print("after update batch loss:", loss_fn.forward(new_prediction, y))

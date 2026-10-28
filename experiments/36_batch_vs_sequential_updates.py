import numpy as np

from src.layers import Dense
from src.losses import MSELoss
from src.optimizers import SGD


# 【目标】比较一次批次更新与两次单样本更新。
# 【输入/输出】X (2,1)、y (2,1)，固定初始 W、b；打印梯度和最终参数。
# 【步骤 1】两组 Dense 均从 W=0、b=0 开始，学习率均为 0.1。
# 【步骤 2】第一组用两个样本的平均损失计算梯度，更新一次。
# 【步骤 3】第二组先用样本 1 更新，再用更新后的参数处理样本 2。
# 【验收】批次更新后的参数仍为 0；逐样本更新的第二次梯度不等于
# 实验 35 中初始参数下样本 2 的梯度，最终参数也与批次更新不同。
# 先预测，再运行；每次反向都紧跟对应前向。

X = np.array([[1.0], [1.0]])
y = np.array([[1.0], [-1.0]])
loss_fn = MSELoss()
optimizer = SGD(learning_rate=0.1)


def make_layer():
    layer = Dense(1, 1)
    layer.W = np.array([[0.0]])
    layer.b = np.array([0.0])
    return layer


def update(layer, inputs, targets):
    prediction = layer.forward(inputs)
    loss = loss_fn.forward(prediction, targets)
    layer.backward(loss_fn.backward(prediction, targets))
    print("  loss:", loss, "dW:", layer.dW, "db:", layer.db)
    optimizer.step(layer)
    print("  after update W:", layer.W, "b:", layer.b)


batch_layer = make_layer()
print("batch update:")
update(batch_layer, X, y)

sequential_layer = make_layer()
print("sequential update, sample 1:")
update(sequential_layer, X[:1], y[:1])
print("sequential update, sample 2:")
update(sequential_layer, X[1:], y[1:])

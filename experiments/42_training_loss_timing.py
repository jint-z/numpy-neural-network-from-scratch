import numpy as np

from src.layers import Dense
from src.losses import MSELoss
from src.optimizers import SGD


# 【当前任务】比较训练过程中各批损失的平均值与更新后的全数据损失。
# 【输入/输出】X、y 均为 (2,1)，批次大小为 1；打印初始全数据 loss、
# 每批更新前的 loss、两批 loss 的按样本数加权平均，以及最终全数据 loss。
# 【步骤 1】固定初始 W=0、b=0 和 SGD 学习率 0.1，计算初始全数据 loss。
# 【步骤 2】按样本 1→2 逐批前向、计算 loss、反向并更新；记录每批
# 更新前的 loss 和样本数，不在前向与对应反向之间插入其他前向。
# 【步骤 3】两次更新后，汇总记录的批次损失，再重新计算全数据 loss。
# 【验收】两批各 1 个样本，加权平均与最终全数据 loss 均有限；比较并
# 解释它们是否相同以及各自对应的参数状态。
# 【后续任务】继续观察训练指标。

X = np.array([[1.0], [1.0]])
y = np.array([[1.0], [-1.0]])
layer = Dense(1, 1)
layer.W = np.array([[0.0]])
layer.b = np.array([0.0])
loss_fn = MSELoss()
optimizer = SGD(learning_rate=0.1)

initial_loss = loss_fn.forward(layer.forward(X), y)
print("initial full-data loss:", initial_loss)
loss_sum = 0.0
sample_count = 0
for i in range(len(X)):
    X_batch, y_batch = X[i:i + 1], y[i:i + 1]
    prediction = layer.forward(X_batch)
    batch_loss = loss_fn.forward(prediction, y_batch)
    batch_size = len(X_batch)
    loss_sum += batch_loss * batch_size
    sample_count += batch_size
    print("sample:", i + 1, "W before:", layer.W, "b before:", layer.b,
          "batch loss before update:", batch_loss)
    layer.backward(loss_fn.backward(prediction, y_batch))
    optimizer.step(layer)

weighted_loss = loss_sum / sample_count
final_loss = loss_fn.forward(layer.forward(X), y)
print("weighted training-batch loss:", weighted_loss)
print("final W:", layer.W, "b:", layer.b, "full-data loss:", final_loss)

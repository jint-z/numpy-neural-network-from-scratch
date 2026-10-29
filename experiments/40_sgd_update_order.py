import numpy as np

from src.layers import Dense
from src.losses import MSELoss
from src.optimizers import SGD


# 【当前任务】观察 SGD 下两个单样本批次的处理顺序对参数的影响。
# 【输入/输出】X (2,1)、y (2,1)；两组初始 W、b 均为 0，分别按
# 样本 1→2 和样本 2→1 更新，打印各步梯度和参数及最终全数据 loss。
# 【步骤 1】两组使用相同样本、初始参数和 SGD 学习率 0.1。
# 【步骤 2】按各自顺序逐样本前向、反向，每个样本更新一次。
# 【步骤 3】比较两组最终参数和全数据平均 loss。
# 【验收】每组更新两次、计算结果有限；观察顺序是否影响最终参数。
# 【后续任务】根据需要继续研究批次训练。

X = np.array([[1.0], [1.0]])
y = np.array([[1.0], [-1.0]])


def train(name, order):
    layer = Dense(1, 1)
    layer.W = np.array([[0.0]])
    layer.b = np.array([0.0])
    loss_fn = MSELoss()
    optimizer = SGD(learning_rate=0.1)
    print("order:", name)
    for i in order:
        X_batch, y_batch = X[i:i + 1], y[i:i + 1]
        prediction = layer.forward(X_batch)
        layer.backward(loss_fn.backward(prediction, y_batch))
        print("  sample:", i + 1, "dW:", layer.dW, "db:", layer.db)
        optimizer.step(layer)
        print("  W:", layer.W, "b:", layer.b)

    loss = loss_fn.forward(layer.forward(X), y)
    print("  final full-data loss:", loss)


train("1 then 2", [0, 1])
train("2 then 1", [1, 0])

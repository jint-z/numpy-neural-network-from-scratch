import numpy as np

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import Adam


# 【当前任务】固定批次组成，只改变批次处理的先后顺序。
# 【输入/输出】X_train (12,2)、y_train (12,)；三个批次大小为 5、5、2；
# 两组训练 10 个 epoch，打印批次索引、累计更新次数和全训练集 loss。
# 【步骤 1】两组使用同一组批次、相同初始参数及训练设置；一组按
# 先前顺序处理，另一组将三个完整批次倒序，每轮重复各自顺序。
# 【步骤 2】每批完成前向、反向后更新两层参数。
# 【步骤 3】比较初始、第 1 轮和第 10 轮的全训练集 loss 与更新次数。
# 【验收】批次组成相同，每轮均更新 3 次，10 轮累计各 30 次；
# 初始 loss 相同，所打印 loss 有限。根据结果讨论，不预设哪组更好。
# 【后续任务】继续分析批次训练的表现。

X_train = np.array([
    [-2.0, -1.0], [-1.5, -1.2], [-2.2, -0.5], [-1.3, -0.7],
    [2.0, -1.0], [1.5, -1.2], [2.2, -0.5], [1.3, -0.7],
    [0.0, 2.0], [-0.5, 1.5], [0.5, 1.5], [0.2, 2.2],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
order = np.random.default_rng(7).permutation(len(X_train))
batches = [order[i:i + 5] for i in range(0, len(order), 5)]


def train(name, batch_sequence):
    np.random.seed(42)
    dense1, relu, dense2 = Dense(2, 3), ReLU(), Dense(3, 3)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    updates = 0

    def training_loss():
        logits = dense2.forward(relu.forward(dense1.forward(X_train)))
        return loss_fn.forward(logits, y_train)

    print("sequence:", name, "batches:", [batch.tolist() for batch in batch_sequence])
    print("epoch: 0", "updates: 0", "train loss:", training_loss())
    for epoch in range(1, 11):
        for indices in batch_sequence:
            X_batch, y_batch = X_train[indices], y_train[indices]
            logits = dense2.forward(relu.forward(dense1.forward(X_batch)))
            loss_fn.forward(logits, y_batch)
            d_logits = loss_fn.backward()
            dA1 = dense2.backward(d_logits)
            dZ1 = relu.backward(dA1)
            dense1.backward(dZ1)
            optimizer.step(dense1)
            optimizer.step(dense2)
            updates += 1

        if epoch in (1, 10):
            print("epoch:", epoch, "updates:", updates, "train loss:", training_loss())


train("forward", batches)
train("reverse", batches[::-1])

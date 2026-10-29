import numpy as np

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import Adam


# 【当前任务】比较批次大小对每轮更新次数与训练损失的影响。
# 【输入/输出】12 个样本 X_train (12,2)、标签 y_train (12,)；
# 分别使用 batch_size=12 和 5，打印初始及第 1、5、10 轮的全训练集 loss，
# 以及每轮和累计更新次数。最后不足一批的样本保留。
# 【步骤 1】两组网络使用相同种子、结构、Adam 学习率和固定样本顺序。
# 【步骤 2】每组训练 10 个 epoch；每批前向、反向后更新两层一次。
# 【步骤 3】在指定轮次全部更新完成后，重新计算整个训练集的平均 loss。
# 【验收】每轮分别更新 1 次和 3 次，10 轮累计 10 次和 30 次；
# 所打印 loss 有限。观察结果，不预设更小批次必然更好。
# 【后续任务】比较数据顺序的影响。

X_train = np.array([
    [-2.0, -1.0], [-1.5, -1.2], [-2.2, -0.5], [-1.3, -0.7],
    [2.0, -1.0], [1.5, -1.2], [2.2, -0.5], [1.3, -0.7],
    [0.0, 2.0], [-0.5, 1.5], [0.5, 1.5], [0.2, 2.2],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
# 两组使用同一个固定顺序；不在每个 epoch 重新打乱。
order = np.random.default_rng(7).permutation(len(X_train))
X_train, y_train = X_train[order], y_train[order]


def train(batch_size):
    np.random.seed(42)
    dense1, relu, dense2 = Dense(2, 3), ReLU(), Dense(3, 3)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    updates = 0

    def training_loss():
        logits = dense2.forward(relu.forward(dense1.forward(X_train)))
        return loss_fn.forward(logits, y_train)

    print("batch size:", batch_size, "epoch: 0", "updates: 0", "train loss:", training_loss())
    for epoch in range(1, 11):
        epoch_updates = 0
        for start in range(0, len(X_train), batch_size):
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            logits = dense2.forward(relu.forward(dense1.forward(X_batch)))
            loss_fn.forward(logits, y_batch)
            d_logits = loss_fn.backward()
            dA1 = dense2.backward(d_logits)
            dZ1 = relu.backward(dA1)
            dense1.backward(dZ1)
            optimizer.step(dense1)
            optimizer.step(dense2)
            epoch_updates += 1
            updates += 1

        # 评估前向发生在全部批次反向完成后；下一批前向会重新建立缓存。
        if epoch in (1, 5, 10):
            print("batch size:", batch_size, "epoch:", epoch,
                  "updates this epoch:", epoch_updates, "total updates:", updates,
                  "train loss:", training_loss())


train(batch_size=12)
train(batch_size=5)

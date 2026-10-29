import numpy as np

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import Adam


# 【当前任务】观察样本顺序如何影响逐批训练。
# 【输入/输出】X_train (12,2)、y_train (12,)；batch_size=5，训练 10 个
# epoch，比较两种固定顺序的初始、第 1 轮和第 10 轮全训练集 loss。
# 【步骤 1】两组从相同初始参数开始，使用相同样本、标签和训练设置；
# 第二组按第一组的逆序取样，每轮均保持各自的顺序。
# 【步骤 2】每批前向、反向后更新两层；每轮保留末尾 2 个样本。
# 【步骤 3】打印各组批次索引、指定轮次 loss 与累计更新次数。
# 【验收】两组每轮都处理同样的 12 个样本、更新 3 次；初始 loss 相同，
# 10 轮累计各更新 30 次，loss 有限。比较结果，不预设哪组更好。
# 【后续任务】结合观察解释训练过程。

X_train = np.array([
    [-2.0, -1.0], [-1.5, -1.2], [-2.2, -0.5], [-1.3, -0.7],
    [2.0, -1.0], [1.5, -1.2], [2.2, -0.5], [1.3, -0.7],
    [0.0, 2.0], [-0.5, 1.5], [0.5, 1.5], [0.2, 2.2],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
order = np.random.default_rng(7).permutation(len(X_train))


def train(name, sample_order):
    np.random.seed(42)
    dense1, relu, dense2 = Dense(2, 3), ReLU(), Dense(3, 3)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    batch_size = 5
    updates = 0

    def training_loss():
        logits = dense2.forward(relu.forward(dense1.forward(X_train)))
        return loss_fn.forward(logits, y_train)

    print("order:", name, "batches:", [sample_order[i:i + batch_size].tolist()
                                        for i in range(0, len(X_train), batch_size)])
    print("epoch: 0", "updates: 0", "train loss:", training_loss())
    for epoch in range(1, 11):
        for start in range(0, len(X_train), batch_size):
            indices = sample_order[start:start + batch_size]
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

        # 整轮更新结束后评估；下一次训练前向会重新建立缓存。
        if epoch in (1, 10):
            print("epoch:", epoch, "updates:", updates, "train loss:", training_loss())


train("forward", order)
train("reverse", order[::-1])

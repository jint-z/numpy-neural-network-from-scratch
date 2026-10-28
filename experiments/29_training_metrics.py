import numpy as np

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import Adam


# 当前任务：观察训练损失和准确率随参数更新的变化。
# 【输入/输出】沿用实验 27 的训练数据：X_train (12,2)、y_train (12,)；
# loss_history、accuracy_history 分别记录 201 个训练集指标。
# 【步骤 1】初始化与实验 27 相同的网络和优化器，从头开始训练。
# 【步骤 2】记录更新前的指标；随后每次更新后再次记录，累计更新 200 次。
# 【步骤 3】打印更新 0、50、100、150、200 次后的 loss 和准确率。
# 【验收】两个记录各有 201 个值，loss 有限、准确率在 [0,1]；
# 结合输出观察两个指标的变化，不要求逐次变化一致。
# 此实验只观察训练集，不用于判断独立评估集上的表现。

# 直接保留实验 27 的数据，避免导入它时重复执行整个训练脚本。
X_train = np.array([
    [-2.0, -1.0], [-1.5, -1.2], [-2.2, -0.5], [-1.3, -0.7],
    [2.0, -1.0], [1.5, -1.2], [2.2, -0.5], [1.3, -0.7],
    [0.0, 2.0], [-0.5, 1.5], [0.5, 1.5], [0.2, 2.2],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

np.random.seed(42)
dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 3)
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = Adam(learning_rate=0.01)

num_updates = 200
loss_history = []
accuracy_history = []

for updates in range(num_updates + 1):
    # 这次前向既用于记录指标，也为接下来的反向传播保存缓存。
    Z1 = dense1.forward(X_train)
    A1 = relu.forward(Z1)
    logits = dense2.forward(A1)
    loss = loss_fn.forward(logits, y_train)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y_train)
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    if updates % 50 == 0:
        print("updates:", updates, "loss:", loss, "accuracy:", accuracy)

    # 最后一次只记录第 200 次更新后的指标，不再额外更新。
    if updates == num_updates:
        break

    d_logits = loss_fn.backward()
    dA1 = dense2.backward(d_logits)
    dZ1 = relu.backward(dA1)
    dense1.backward(dZ1)
    optimizer.step(dense1)
    optimizer.step(dense2)

print("history lengths:", len(loss_history), len(accuracy_history))

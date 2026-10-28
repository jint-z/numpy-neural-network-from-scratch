import runpy

import numpy as np

from src.losses import SoftmaxCrossEntropyLoss


# 当前任务：比较两个单样本对参数梯度的贡献。
# 【输入/输出】两个点各为 (1,2)，假设类别标签各为 (1,)；输出第一层
# Z1、A1、dZ1 (1,3)，两层 dW、db，以及单样本损失。
# 【步骤 1】运行实验 29，从头训练 200 次，取出训练后的网络。
# 【步骤 2】逐个对两个点完成前向、计算损失、反向传播；不更新参数。
# 每个样本的前向与反向必须相邻完成，避免缓存被下一个样本覆盖。
# 【步骤 3】分别打印各层梯度，比较 ReLU 全关与部分激活的情况。
# 【验收】第一个点的 Z1 严格小于 0，A1 和 dZ1 全为 0；观察两层 dW、db。
# 第二个点用于对照。标签 2 是假设值，不代表这些无标签点的真实类别。

trained = runpy.run_module("experiments.29_training_metrics")
dense1 = trained["dense1"]
relu = trained["relu"]
dense2 = trained["dense2"]
loss_fn = SoftmaxCrossEntropyLoss()

points = np.array([[0.7, 1.25], [0.7, 0.5]])
assumed_label = np.array([2])

for point in points:
    # 保留批次维度：一个样本也需要形状 (1,2)。
    X = point.reshape(1, 2)
    Z1 = dense1.forward(X)
    A1 = relu.forward(Z1)
    logits = dense2.forward(A1)
    loss = loss_fn.forward(logits, assumed_label)

    # 先完成所有反向传播；本实验只检查梯度，不调用优化器。
    d_logits = loss_fn.backward()
    dA1 = dense2.backward(d_logits)
    dZ1 = relu.backward(dA1)
    dense1.backward(dZ1)

    print("point:", point, "assumed label:", assumed_label[0], "loss:", loss)
    print("  Z1:", Z1, "A1:", A1, "dZ1:", dZ1)
    print("  dense1 dW:", dense1.dW, "db:", dense1.db)
    print("  dense2 dW:", dense2.dW, "db:", dense2.db)

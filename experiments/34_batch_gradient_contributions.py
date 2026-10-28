import runpy

import numpy as np

from src.losses import SoftmaxCrossEntropyLoss


# 【目标】比较两个单样本梯度与这两个样本组成的批次梯度。
# 【输入/输出】points (2,2)、假设标签 (2,)；比较两层 dW 和 db。
# 【步骤 1】运行实验 29 训练网络，后续只计算梯度，不更新参数。
# 【步骤 2】分别计算两个单样本梯度，每次前向后立即反向，并复制梯度。
# 【步骤 3】计算两样本批次梯度，与两个单样本梯度的平均值比较。
# 【验收】四组梯度分别近似相等；先预测原因，再运行观察并解释。
# 标签是用于实验的假设值，不代表这些无标签点的真实类别。

trained = runpy.run_module("experiments.29_training_metrics")
dense1, relu, dense2 = trained["dense1"], trained["relu"], trained["dense2"]
loss_fn = SoftmaxCrossEntropyLoss()
points = np.array([[0.7, 1.25], [0.7, 0.5]])
labels = np.array([2, 2])


def gradients(X, y):
    Z1 = dense1.forward(X)
    A1 = relu.forward(Z1)
    logits = dense2.forward(A1)
    loss_fn.forward(logits, y)
    d_logits = loss_fn.backward()
    dA1 = dense2.backward(d_logits)
    dZ1 = relu.backward(dA1)
    dense1.backward(dZ1)
    # 后续 backward 会覆盖层上的梯度，因此这里保存副本。
    return [dense1.dW.copy(), dense1.db.copy(), dense2.dW.copy(), dense2.db.copy()]


first = gradients(points[:1], labels[:1])
second = gradients(points[1:], labels[1:])
batch = gradients(points, labels)

for name, g1, g2, batch_gradient in zip(
    ("dense1 dW", "dense1 db", "dense2 dW", "dense2 db"), first, second, batch
):
    average = (g1 + g2) / 2
    print(name, "batch:\n", batch_gradient)
    print("  individual average:\n", average)
    print("  close:", np.allclose(batch_gradient, average))

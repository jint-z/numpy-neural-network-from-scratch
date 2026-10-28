import runpy

import numpy as np

from src.layers import Softmax


# 当前任务：检查实验 31 中部分不同点得到相同概率的原因。
# 【输入/输出】points (4,2)；Z1、A1、logits、probabilities 均为 (4,3)。
# 【步骤 1】运行实验 29，从头训练 200 次，取出训练后的网络。
# 【步骤 2】对选取的 4 个点依次计算第一层 Dense、ReLU、最后一层 Dense
# 和 Softmax 的输出；逐点打印这些中间结果。
# 【步骤 3】比较后 3 个点的 ReLU 输出是否相同，并对照它们的概率。
# 【验收】输出形状符合约定，每行概率和接近 1；根据实际结果解释
# 从哪一层开始结果相同。不预设所有不同输入必然输出不同概率。

# 新点只用于观察，不参与参数更新。
trained = runpy.run_module("experiments.29_training_metrics")
dense1 = trained["dense1"]
relu = trained["relu"]
dense2 = trained["dense2"]

points = np.array([
    [0.7, 0.5],
    [0.7, 0.75],
    [0.7, 1.25],
    [0.7, 2.0],
])

Z1 = dense1.forward(points)
A1 = relu.forward(Z1)
logits = dense2.forward(A1)
probabilities = Softmax().forward(logits)

print("shapes:", points.shape, Z1.shape, A1.shape, logits.shape, probabilities.shape)
print("probability row sums:", probabilities.sum(axis=1))
for i in range(len(points)):
    print("point:", points[i])
    print("  Z1:", Z1[i], "A1:", A1[i])
    print("  logits:", logits[i], "probabilities:", probabilities[i])

# 与第 2 个点比较后两个点，浮点结果使用 allclose 判断近似相等。
print("last three ReLU outputs close:", np.allclose(A1[1:], A1[1]))

import runpy

import numpy as np

from src.layers import Softmax


# 当前任务：观察已训练模型对新位置的预测。
# 【输入/输出】probe_points (6,2)，每行是一个新点；输出 logits、
# probabilities (6,3) 和 predictions (6,)，以及每行前两名概率的差。
# 【步骤 1】运行实验 29，从头训练网络 200 次，并取出训练后的层。
# 【步骤 2】仅对新点做前向，计算类别分数、Softmax 概率及预测类别。
# 【步骤 3】逐点打印坐标、三个类别的概率、预测和前两名概率差。
# 【验收】概率形状 (6,3)，预测形状 (6,)，概率每行和接近 1；
# 对照结果观察哪些点的前两名较接近。没有真实标签，不能计算准确率
# 或断定预测对错；概率差也不能单独证明模型判断可靠。

# run_module 会完整运行实验 29，先输出训练过程；新点不参与训练。
trained = runpy.run_module("experiments.29_training_metrics")
dense1 = trained["dense1"]
relu = trained["relu"]
dense2 = trained["dense2"]

# 包含靠近不同训练区域的点，也包含区域之间的点；不预设真实类别。
probe_points = np.array([
    [-1.7, -0.9],
    [1.7, -0.9],
    [0.1, 1.7],
    [-0.7, 0.2],
    [0.7, 0.2],
    [0.0, 0.0],
])

logits = dense2.forward(relu.forward(dense1.forward(probe_points)))
probabilities = Softmax().forward(logits)
predictions = np.argmax(probabilities, axis=1)

# 每行从小到大排序：最后两个元素是最大和第二大的概率。
sorted_probabilities = np.sort(probabilities, axis=1)
probability_gaps = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

print("probe shapes:", probe_points.shape, probabilities.shape, predictions.shape)
print("probability row sums:", probabilities.sum(axis=1))
for i in range(len(probe_points)):
    print("point:", probe_points[i], "probabilities:", probabilities[i],
          "prediction:", predictions[i], "top-two gap:", probability_gaps[i])

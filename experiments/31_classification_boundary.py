import runpy

import numpy as np

from src.layers import Softmax


# 当前任务：沿一条线观察模型预测类别的变化。
# 【输入/输出】固定第一个特征为 0.7，让第二个特征从 -1 到 2 等间距变化，
# 生成 points (13,2)；得到 probabilities (13,3)、predictions (13,)。
# 【步骤 1】运行实验 29，从头训练 200 次，取出训练后的网络。
# 【步骤 2】将 13 个点仅用于前向预测，打印坐标、各类别概率和预测类别。
# 【步骤 3】比较相邻预测，打印预测类别发生变化的相邻点区间。
# 【验收】点、概率、预测的形状分别为 (13,2)、(13,3)、(13,)；
# 每行概率和接近 1。相邻点预测不同时，只能确定变化发生在这两点之间，
# 不能从这组离散采样确定精确位置；点没有真实标签，不评估对错。

# 与实验 30 一样，重新运行实验 29 会先打印训练过程。
trained = runpy.run_module("experiments.29_training_metrics")
dense1 = trained["dense1"]
relu = trained["relu"]
dense2 = trained["dense2"]

second_feature = np.linspace(-1.0, 2.0, 13)
first_feature = np.full_like(second_feature, 0.7)
points = np.column_stack((first_feature, second_feature))

logits = dense2.forward(relu.forward(dense1.forward(points)))
probabilities = Softmax().forward(logits)
predictions = np.argmax(probabilities, axis=1)

print("shapes:", points.shape, probabilities.shape, predictions.shape)
print("probability row sums:", probabilities.sum(axis=1))
for i in range(len(points)):
    print("point:", points[i], "probabilities:", probabilities[i],
          "prediction:", predictions[i])

# 比较第 i 个点与第 i+1 个点；只报告相邻采样点之间的类别变化。
change_indices = np.flatnonzero(predictions[:-1] != predictions[1:])
print("number of changed intervals:", len(change_indices))
for i in change_indices:
    print("changed interval:", points[i], "->", points[i + 1],
          "classes:", predictions[i], "->", predictions[i + 1])

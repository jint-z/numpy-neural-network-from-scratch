import numpy as np

from src.losses import CrossEntropyLoss


# 当前任务：观察分类损失和准确率为什么可能不同步。
# 【输入/输出】两组预测概率均为 (2,3)，真实类别 y_true 为 (2,)；
# 每组得到预测类别 (2,)、准确率标量和平均交叉熵损失标量。
# 【步骤】分别计算两组的预测类别、准确率和平均损失，再进行比较。
# 【验收】先预测两组的预测类别、准确率及损失变化，再运行并解释结果。
y_true = np.array([0, 2])
probabilities_before = np.array([
    [0.40, 0.35, 0.25],
    [0.20, 0.50, 0.30],
])
probabilities_after = np.array([
    [0.80, 0.10, 0.10],
    [0.30, 0.40, 0.30],
])

loss_fn = CrossEntropyLoss()

# 每行选概率最大的类别；布尔比较的平均值就是预测正确的比例。
predictions_before = np.argmax(probabilities_before, axis=1)
accuracy_before = np.mean(predictions_before == y_true)
loss_before = loss_fn.forward(probabilities_before, y_true)

predictions_after = np.argmax(probabilities_after, axis=1)
accuracy_after = np.mean(predictions_after == y_true)
loss_after = loss_fn.forward(probabilities_after, y_true)

print("labels:", y_true)
print("before:", predictions_before, "accuracy:", accuracy_before, "loss:", loss_before)
print("after: ", predictions_after, "accuracy:", accuracy_after, "loss:", loss_after)

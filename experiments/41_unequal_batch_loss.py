import numpy as np

from src.losses import MSELoss


# 【当前任务】比较不等大小批次的平均损失与全数据平均损失。
# 【输入/输出】predictions、targets 均为 (7,1)，分为大小 5 和 2 的两批；
# 打印两批 loss、直接平均、按样本数加权平均及全数据 loss。
# 【步骤 1】用 MSELoss 计算前 5 个样本和后 2 个样本各自的平均损失。
# 【步骤 2】将两个批次 loss 直接平均，再分别乘以样本数求加权平均。
# 【步骤 3】直接计算全部 7 个样本的 loss，并比较两种汇总结果。
# 【验收】加权平均与全数据 loss 一致；观察直接平均是否一致。
# 所有预测使用同一组固定数值，实验不训练、不更新参数。
# 【后续任务】继续观察训练指标的计算。

predictions = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [3.0], [3.0]])
targets = np.zeros((7, 1))
loss_fn = MSELoss()

first_loss = loss_fn.forward(predictions[:5], targets[:5])
second_loss = loss_fn.forward(predictions[5:], targets[5:])
simple_average = (first_loss + second_loss) / 2
weighted_average = (first_loss * 5 + second_loss * 2) / 7
full_loss = loss_fn.forward(predictions, targets)

print("batch sizes:", 5, 2, "batch losses:", first_loss, second_loss)
print("simple average:", simple_average)
print("weighted average:", weighted_average)
print("full-data loss:", full_loss)
print("weighted matches full:", np.allclose(weighted_average, full_loss))

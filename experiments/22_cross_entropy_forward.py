import numpy as np


# 三个样本，每行是三个类别的预测概率，类别编号为 0、1、2。
probabilities = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.3, 0.6],
    [0.2, 0.3, 0.5],
])

# 每个样本的真实类别编号；shape 为 (N,)，不是回归目标 (N,1)。
y_true = np.array([0, 2, 1])

print("probabilities:")
print(probabilities)
print("probabilities shape:", probabilities.shape)
print("y_true:", y_true)
print("y_true shape:", y_true.shape)


# 任务 1：取出每个样本真实类别的预测概率。
# 注意：第三个样本概率最大的类别并不是真实类别，不要直接取行最大值。
# 先在注释里预测三个数及结果 shape，再实现 correct_probabilities。
# 可以先使用循环；也可以用 np.arange 生成行索引，配对 y_true 列索引。
# 打印 correct_probabilities 及其 shape。

N = probabilities.shape[0]
correct_probabilities = np.zeros((N,1),dtype=float)
for arrange in range(N):
    index_col = y_true[arrange]
    correct_probabilities [arrange][0] = probabilities[arrange][index_col]
#预测 三个数：0.7,0.6,0.3  shape：(3,1)
print("correct_probablities.shape:",correct_probabilities.shape)
print("correct_probabilities:",correct_probabilities)





# 任务 2：计算交叉熵前向损失。
# 每个样本的 loss = -ln(该样本真实类别的预测概率)。
# 用 np.log 计算 sample_losses，再求平均得到标量 loss，分别打印。
# 解释：真实类别概率变大时，该样本损失怎样变化，为什么？
# 本例真实类别概率均大于 0；概率为 0 的数值处理留到后续封装讨论。

sample_losses = -np.log(correct_probabilities)
loss = np.mean(sample_losses)
print("sample_losses:",sample_losses)
print("loss:",loss)
#变小，原因如下；
#真实概率变大，对数变大，负对数变小，也就是sample_losses变小，则loss变小

from src.losses import CrossEntropyLoss

loss_fn = CrossEntropyLoss()
class_loss = loss_fn.forward(probabilities, y_true)
print("class loss:", class_loss)
print("一致吗:", np.allclose(class_loss, loss))
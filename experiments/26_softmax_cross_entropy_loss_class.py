import numpy as np
from src.losses import SoftmaxCrossEntropyLoss

logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0],
    [0.2, 0.3, 0.5],
])
y_true = np.array([0, 2, 1])
extreme_logits = np.array([[1000.0, -1000.0]])
extreme_y_true = np.array([1])

print("ordinary input shapes:", logits.shape, y_true.shape)
print("extreme input shapes:", extreme_logits.shape, extreme_y_true.shape)


# 当前任务：实验 26，调用与验证写在下方（步骤 4）。
#
# 【目标】验证 src/losses.py 中新增的 SoftmaxCrossEntropyLoss。
# 类实现的步骤 1–3 提示位于 src/losses.py 的待实现位置。
# 【输入/输出】普通输入 logits (3,3)、y_true (3,)；极端输入
# extreme_logits (1,2)、extreme_y_true (1,)。分别得到 loss 标量与 d_logits。
#
# 【步骤 4：调用验证】
# 在本文件导入并实例化该类，先普通输入 forward，再立即 backward，打印
# loss、d_logits、shape、每行梯度和及 self.probabilities 的行和。
# 再极端输入 forward，立即 backward，打印对应的 loss 与 d_logits。
# 普通反向完成后再做极端前向，避免覆盖普通输入的缓存。
# 这里标签由 forward 参数传入，不修改全局 y_true。
#
# 【验收】普通 loss 约 0.9881556805222247，d_logits shape (3,3)，
# 每行梯度和近似 0，缓存概率行和近似 1。
# 极端 loss 为 2000（且有限），d_logits shape (1,2)，接近 [[1.0,-1.0]]。
# 实现留给用户完成；此文件当前仅打印输入形状，不代表类实现已通过。

SoftmaxCrossEntropyloss = SoftmaxCrossEntropyLoss()
loss = SoftmaxCrossEntropyloss.forward(logits,y_true)
d_logit = SoftmaxCrossEntropyloss.backward()
print("loss:",loss)
print("d_logit:",d_logit)
print("d_logit.shape:",d_logit.shape)
Buffer_probability_row_sum = np.sum(SoftmaxCrossEntropyloss.probabilities , axis = 1)
gradient_row_sums = np.sum(d_logit,axis=1)
print("Buffer_probability_row_sum:",Buffer_probability_row_sum)
print("gradient_row_sumsgradient_row_sums:",gradient_row_sums)
ex_loss = SoftmaxCrossEntropyloss.forward(extreme_logits,extreme_y_true)
ex_d_logit = SoftmaxCrossEntropyloss.backward()
print("ex_loss:",ex_loss)
print("ex_d_logit",ex_d_logit)

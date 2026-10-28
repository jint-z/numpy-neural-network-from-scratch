import numpy as np

from src.layers import Softmax
from src.losses import CrossEntropyLoss


# 三个样本、三个类别。这里的输入是 logits，不是概率。
logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0],
    [0.2, 0.3, 0.5],
])
y_true = np.array([0, 2, 1])
N = logits.shape[0]

softmax = Softmax()
probabilities = softmax.forward(logits)
loss_fn = CrossEntropyLoss()
loss = loss_fn.forward(probabilities, y_true)

print("probabilities:")
print(probabilities)
print("loss:", loss)
print("smallest probability:", np.min(probabilities))


# 任务：计算批平均交叉熵对 logits 的梯度，命名为 d_logits。
# 单个样本：真实类别位置的梯度为 p_j - 1，其余位置为 p_j。
# 批平均：所有样本的上述梯度再除以样本数 N。
# 1. 用 probabilities.copy() 创建独立的梯度数组，保留原概率。
# 2. 用 range(N) 遍历样本，在该行真实类别列的位置减去 1。
# 3. 整个梯度数组除以 N（不是 probabilities.size）。
# 4. 打印 d_logits、shape 和每行的和（sum(axis=1)）。
# 先预测：真实类别位置和其他位置各是什么符号？每行的和是多少？
# 补充解释：为什么上面的结果可以作为最后一层 Dense 的上游梯度？
# 不知道
#预测：真实类别为负，其他为正，每行的和应该是0 
d_logits = probabilities.copy()
for i in range(N):
    d_logits[i,y_true[i]] -= 1
d_logits/=N
print("d_logits:",d_logits)
print("d_logits.shape:",d_logits.shape)
print("sum(axis=1):",np.sum(d_logits,axis=1))




# 本例概率远大于 1e-12，前向的截断不生效。
# 本任务推导的是 Softmax + 交叉熵组合对 logits 的梯度。
# 它不是交叉熵对概率的梯度，也不是通用 Softmax.backward。
# 极小概率的稳定组合损失、数值梯度检查及类封装在后续实验完成。

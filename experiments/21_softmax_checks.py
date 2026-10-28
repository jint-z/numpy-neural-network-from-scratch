import numpy as np

from src.layers import Softmax


# 两个样本、三个类别。每一行是一个样本的类别分数（logits）。
logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0],
])

softmax = Softmax()
probabilities = softmax.forward(logits)

print("logits:")
print(logits)
print("probabilities:")
print(probabilities)
print("output shape:", probabilities.shape)
print("row sums:", probabilities.sum(axis=1))


# 任务 1：先预测形状，再在这里补代码验证。

# 用与你的 Softmax.forward 相同的步骤计算 max_values、shifted_logits、
# exp_values、sum_exp，并打印四者的 shape。
# 解释：axis=1 聚合了什么？keepdims=True 保留了什么，为什么需要它？
max_values = np.max(logits,axis=1,keepdims=True)
shifted_logits = logits - max_values
exp_value = np.exp(shifted_logits)
sum_exp = np.sum(exp_value,axis=1,keepdims = True)
#预测 (1,3)
print("max_values.shape:",max_values.shape)
#预测 (2,3)
print("shifted_logits.shape:",shifted_logits.shape)
#预测 (2,3)
print("exp_value.shape:",exp_value.shape)
#预测 (1,3)
print("sum_exp.shape:",sum_exp.shape)
#解释: axis=1 聚合了列 keepdims 保留了维度 方便做矩阵逐个元素减法和除法

# 任务 2：每个样本的所有类别分数一起平移。
row_offsets = np.array([[1000.0], [-1000.0]])
shifted_logits = logits + row_offsets
# 补代码：对 shifted_logits 调用 forward，打印概率与是否全部有限。
# 用 np.allclose 比较它与原 probabilities，先预测结果再运行。
# 解释：为什么允许两行使用不同的平移量？

# 预测: 概率不变
# 代码
new_probabilities = softmax.forward(shifted_logits)
print("new_probablities:",new_probabilities)
# 预测: True
flag = np.allclose(probabilities,new_probabilities)
print("the same? :",flag)
print("all finite:", np.isfinite(new_probabilities).all())
#解释: 概率的计算本质是把行作为单位，所以不同行使用不同偏移量并不影响行内的计算结果


# 任务 3：只有一个类别分数。
single_output_logits = np.array([[2.0], [7.0]])
# 补代码：调用 forward，打印结果，先预测再运行。
single_output_probablities = softmax.forward(single_output_logits)
# 预测：shape (2,1) 
print("single_output_probablities:",single_output_probablities)
# 解释：如果给现有单输出回归网络接上 Softmax，会发生什么？
# 无论最后Dense输出的分数是多少，softmax.forword 的结果都是1，这是因为每一行只有一个分数，无论这个分数是多少，都在这一行占比100%

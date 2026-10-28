import numpy as np

from src.losses import CrossEntropyLoss


# 每行仍是合法的概率分布，但第二个样本的真实类别概率为 0。
probabilities = np.array([
    [0.7, 0.2, 0.1],
    [1.0, 0.0, 0.0],
    [0.2, 0.3, 0.5],
])
y_true = np.array([0, 1, 2])

loss_fn = CrossEntropyLoss()
loss = loss_fn.forward(probabilities, y_true)
print("loss with a zero true-class probability:", loss)
print("finite loss:", np.isfinite(loss))

# 真实类别概率为 1 时，损失应仍为 0。
certain_probabilities = np.array([[1.0, 0.0, 0.0]])
certain_label = np.array([0])
certain_loss = loss_fn.forward(certain_probabilities, certain_label)
print("loss with true-class probability 1:", certain_loss)


# 加入保护前，首次运行出现 divide by zero 警告和 inf 是预期现象。
# 任务：在 CrossEntropyLoss.forward 中完成真实类别概率选取后、取对数前，
# 使用 np.clip 将选出的概率限制在 [1e-12, 1.0]，再计算原来的损失。
# 修改后运行本实验：零概率示例的损失应有限，概率 1 示例应仍为 0。
# 再运行 22：原来的普通概率示例应保持约 0.690491124。
# 用自己的话解释为什么保护必须发生在 np.log 之前。

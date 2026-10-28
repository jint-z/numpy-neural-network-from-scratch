import numpy as np
class MSELoss:
    def forward(self, y_pred, y_true):
        # 返回一个标量 loss
        loss = np.mean((y_pred-y_true)**2)
        return loss

    def backward(self, y_pred, y_true):
        # 返回 dL/dy_pred
        return 2*(y_pred-y_true)/y_pred.size


class CrossEntropyLoss:
    def forward(self,y_pred,y_true):
        N = y_pred.shape[0]
        correct_probabilities = np.zeros((N,1),dtype=float)
        for arrange in range(N):
            index_col = y_true[arrange]
            correct_probabilities[arrange,0] = y_pred[arrange][index_col]
        correct_probabilities = np.clip(correct_probabilities, 1e-12, 1.0)
        sample_losses = -np.log(correct_probabilities)
        loss = np.mean(sample_losses)
        return loss


# 当前任务：实验 26，类实现写在下方（步骤 1–3）。
#
# 【目标】新增 SoftmaxCrossEntropyLoss，将实验 25 的稳定前向与组合梯度
# 封装在一起；保留上面的 MSELoss、CrossEntropyLoss。
# 【输入/输出】forward(self, logits, y_true) 接收浮点 logits (N,C) 和
# 整数标签 (N,)，返回平均损失标量；backward(self) 返回 d_logits (N,C)。
# backward 使用最近一次 forward 的缓存，无须再传入标签或分数。
#
# 【步骤 1：稳定前向】
# 将实验 25 的 f_loss 逻辑搬进 forward，使用传入的 logits 和 y_true。
# 变量沿用 max_values、shifted_logits、log_sum_exp、sample_losses、loss。
# 逐行减最大值，再计算 exp_values、sum_exp（axis=1, keepdims=True）。
# log_sum_exp 为 np.log(sum_exp)，逐行减真实类别的 shifted_logits，再求平均。
# 此类直接接收 logits，不调用旧的截断概率交叉熵计算损失。
#
# 【步骤 2：保存缓存】
# forward 内保存 self.probabilities = exp_values / sum_exp，shape (N,C)；
# 保存 self.y_true 为传入标签的副本，self.N 为 logits.shape[0]。
# 这些缓存应在每次 forward 更新；所有保存操作放在 return loss 之前。
#
# 【步骤 3：组合反向】
# 实现 backward(self)：将 self.probabilities.copy() 命名为 d_logits。
# 用 range(self.N) 遍历，每行 self.y_true 指定的类别列减 1，
# 再除以 self.N，并 return d_logits；不要修改缓存的概率。
# 返回值已经是损失对 logits 的梯度，可以直接传给最后一层 Dense.backward。
#
# 【验收】完成类后，到 experiments/26_softmax_cross_entropy_loss_class.py
# 按步骤 4 调用验证；具体输入与预期结果写在该实验的代码位置。

class SoftmaxCrossEntropyLoss:
    def backward(self):
        d_logits = self.probabilities.copy()
        for i in range (self.N):
            d_logits[i][self.y_true[i]]-=1
        d_logits /= self.N
        return d_logits
    def forward(self,logits,y_true):
        N = logits.shape[0]
        max_values = np.max(logits,axis=1,keepdims=True)
        shifted_logits = logits - max_values
        exp_values = np.exp(shifted_logits)
        sum_exp = np.sum(exp_values,axis=1,keepdims=True)
        log_sum_exp =np.log(sum_exp)
        sample_losses = np.zeros_like(log_sum_exp)
        for arrange in range(N):
            sample_losses[arrange,0] =log_sum_exp[arrange,0]-shifted_logits[arrange,y_true[arrange]]
        loss = np.mean(sample_losses)
        self.probabilities = exp_values / sum_exp
        self.y_true = y_true.copy()
        self.N = logits.shape[0]
        return loss 
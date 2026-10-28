import numpy as np

from src.layers import Softmax
from src.losses import CrossEntropyLoss


# 沿用实验 17 的中心差分方法，不必重写已完成的检查工具。
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        original = x[i]
        x[i] = original + eps
        loss_plus = f(x)
        x[i] = original - eps
        loss_minus = f(x)
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
        x[i] = original
    return grad


logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0],
    [0.2, 0.3, 0.5],
])
y_true = np.array([0, 2, 1])
probabilities = Softmax().forward(logits)
reference_loss = CrossEntropyLoss().forward(probabilities, y_true)
print("ordinary-input reference loss:", reference_loss)


# 已完成任务 1：稳定交叉熵前向。保留实现和验收输出，供任务 2 使用。
#
# 【目标】定义 f_loss(z)，直接从 logits 计算平均交叉熵。
# 【输入】z：浮点类别分数，shape (N,C)；使用文件已有的 y_true，shape (N,)。
# 【输出】loss：平均损失标量，通过 return 返回。
# 【公式】每行损失 = log(sum(exp(平移后分数))) - 真实类别的平移后分数。
#
# 【步骤 1：逐行平移】
# 从参数 z.shape[0] 获取 N。
# max_values：逐行最大值，使用 axis=1、keepdims=True，shape (N,1)。
# shifted_z：z 减去 max_values，shape (N,C)。
#
# 【步骤 2：逐行对数指数和】
# log_sum_exp：先对 shifted_z 取 np.exp，再沿 axis=1 求 np.sum，
# 保持 keepdims=True，最后取 np.log；结果 shape (N,1)。
#
# 【步骤 3：逐样本损失】
# 用浮点零数组创建 sample_losses，shape (N,1)。
# 用 range(N) 遍历样本；第 i 行损失为 log_sum_exp[i,0]
# 减去 shifted_z 中第 i 行、y_true[i] 列的分数。
# 将这个差写入 sample_losses[i,0]。这里选取的是分数，不是概率。
#
# 【步骤 4：平均并返回】
# loss：对 sample_losses 求平均，得到标量；return loss。
#
# 【验收】函数外打印 f_loss(logits)，应约为 0.9881556805222247，
# 并用 np.allclose 与 reference_loss 比较，结果应为 True。
# 当前直接从 logits 计算损失，不调用概率截断。

def f_loss(z):
    N = z.shape[0]
    max_values = np.max(z,axis=1,keepdims=True)
    shifted_z = z - max_values
    log_sum_exp =np.log(np.sum(np.exp(shifted_z),axis=1,keepdims=True))
    sample_losses = np.zeros_like(log_sum_exp)
    for arrange in range(N):
        sample_losses[arrange,0] =log_sum_exp[arrange,0]-shifted_z[arrange,y_true[arrange]]
    loss = np.mean(sample_losses)
    return loss 
#验收
loss = f_loss(logits)
print("f_loss(logits):",loss)
print("The same as reference_loss :",np.allclose(loss,reference_loss))





# 已完成任务 2：普通输入的数值梯度检查。保留实现和验收输出。
#
# 【目标】比较手写分析梯度与中心差分梯度，核对批平均损失对 logits 的导数。
# 【输入】已有 logits 和 probabilities：shape (N,C)；固定 y_true：shape (N,)。
# 已有 f_loss(z) 返回稳定平均损失；numerical_gradient 是已有的检查函数。
# 【输出】analytical_gradient、numerical_grad、absolute_error：shape (N,C)；
# max_error：最大绝对误差标量；gradient_check_passed：比较结果布尔值。
#
# 【步骤 1：分析梯度】
# 沿用实验 24：analytical_gradient 是 probabilities.copy() 的独立副本。
# 用 range(N) 遍历样本，每行真实类别列减 1，再将整个数组除以样本数 N。
# 样本数由 logits.shape[0] 获取，不除以 logits.size。
analytical_gradient = probabilities.copy()
N = probabilities.shape[0]
for i in range(N):
    analytical_gradient[i,y_true[i]] -= 1
analytical_gradient /= N
#
# 【步骤 2：数值梯度】
# 调用 numerical_gradient(f_loss, logits.copy())，结果命名为 numerical_grad。
# 使用工具默认 eps=1e-5；扰动的是 logits，y_true 在所有扰动中保持不变。
# 工具会临时修改输入，所以传入副本；不要用结果覆盖工具函数的名字。
numerical_grad = numerical_gradient(f_loss,logits.copy())
# 【步骤 3：误差】
# absolute_error：两个梯度相减后，用 np.abs 逐元素取绝对值。
# max_error：用 np.max 取 absolute_error 中的最大值。
# 打印两个梯度、absolute_error 和 max_error。
absolute_error = np.abs(analytical_gradient-numerical_grad)
max_error = np.max(absolute_error)
print("numerical_gradient:",numerical_grad)
print("analytical_gradient:",analytical_gradient)
print("absolute_error:",absolute_error)
print("max_error:",max_error)
# 【步骤 4：判断】
# gradient_check_passed：用 np.allclose 比较 analytical_gradient 与 numerical_grad，
# 指定 atol=1e-7、rtol=1e-5；打印比较结果。
print("gradient_check_passed:",np.allclose(numerical_grad,analytical_gradient,rtol=1e-5,atol=1e-7))
# 【验收】两个梯度均为 (N,C)，max_error 应很小，gradient_check_passed 为 True。
# 浮点运算有舍入误差，不要求误差严格等于 0。


# 已完成任务 3：极端输入验证。保留实现与验收输出。
#
# 【目标】验证极端 logits 下，稳定损失仍有限，分析梯度与数值梯度一致。
# 【输入】extreme_logits = np.array([[1000.0, -1000.0]])，shape (1,2)；
# 真实类别为 1，标签 shape (1,)。沿用 f_loss 和 numerical_gradient。
# 【输出】extreme_loss：标量；两个 extreme 梯度：shape (1,2)；
# extreme_max_error：误差标量；extreme_check_passed：布尔值。
#
# 【步骤 1：输入与标签】
# 创建 extreme_logits。用 original_y_true 保存当前 y_true 的副本，
# 再将 y_true 设为 np.array([1])，以匹配 f_loss 使用的固定标签。
# 用 Softmax().forward(extreme_logits) 得到 extreme_probabilities。
# 不改动普通输入的 logits 或 probabilities。
extreme_logits = np.array([[1000.0, -1000.0]])
original_y_true = y_true.copy()
y_true = np.array([1])
softmax = Softmax()
extreme_probabilities = softmax.forward(extreme_logits)

# 【步骤 2：稳定损失】
# 用 f_loss(extreme_logits) 得到 extreme_loss，打印数值及 np.isfinite 的结果。
# 此处用直接由 logits 计算的稳定损失，不用概率截断后的损失作数值目标。
extreme_loss = f_loss(extreme_logits)
print("extreme_loss:",extreme_loss)
print("isfinite ?",np.isfinite(extreme_loss))
# 【步骤 3：两个梯度】
# extreme_analytical_gradient：复制 extreme_probabilities，按真实类别减 1，
# 再除以 extreme_logits.shape[0]；沿用任务 2 的方法。
# extreme_numerical_grad：调用 numerical_gradient(f_loss, extreme_logits.copy())。
# 使用默认 eps=1e-5；两个梯度计算期间 y_true 保持为 np.array([1])。
extreme_analytical_gradient = extreme_probabilities.copy()
for i in range(extreme_logits.shape[0]):
    extreme_analytical_gradient[i,y_true[i]] -= 1
extreme_analytical_gradient /=  extreme_logits.shape[0]
extreme_numerical_grad = numerical_gradient(f_loss,extreme_logits.copy())
# 【步骤 4：误差与判断】
# 两个梯度相减、取绝对值和最大值，得到 extreme_max_error。
# 用 np.allclose 比较两个梯度，atol=1e-7、rtol=1e-5，得到 extreme_check_passed。
# 打印两个梯度、extreme_max_error 和 extreme_check_passed。
# 全部计算完成后，将 y_true 恢复为 original_y_true。
extreme_max_error = np.max(np.abs(extreme_numerical_grad - extreme_analytical_gradient))
extreme_check_passed = np.allclose(extreme_numerical_grad,extreme_analytical_gradient,rtol=1e-5,atol=1e-7)
print("extreme_numerical_gradient:",extreme_numerical_grad)
print("extreme_analytical_gradient:",extreme_analytical_gradient)
print("extreme_max_error:",extreme_max_error)
print("extreme_check_passed:",extreme_check_passed)
y_true = original_y_true
# 【验收】extreme_loss 接近 2000 且有限；两个梯度接近 [[1.0,-1.0]]；
# extreme_max_error 很小，extreme_check_passed 为 True。

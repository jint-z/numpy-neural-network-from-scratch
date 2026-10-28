import numpy as np

from src.layers import Dense, ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import Adam


# 三个类别分别分布在左下、右下和上方；每个样本有两个特征。
# 训练数据每类 4 个样本，标签是类别索引 0、1、2。
X_train = np.array([
    [-2.0, -1.0], [-1.5, -1.2], [-2.2, -0.5], [-1.3, -0.7],
    [ 2.0, -1.0], [ 1.5, -1.2], [ 2.2, -0.5], [ 1.3, -0.7],
    [ 0.0,  2.0], [-0.5,  1.5], [ 0.5,  1.5], [ 0.2,  2.2],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# 独立评估数据每类 2 个样本；只用来评估，不用于参数更新。
X_test = np.array([
    [-1.8, -0.8], [-1.2, -1.1],
    [ 1.8, -0.8], [ 1.2, -1.1],
    [-0.3,  1.8], [ 0.4,  1.9],
])
y_test = np.array([0, 0, 1, 1, 2, 2])

print("train shapes:", X_train.shape, y_train.shape)
print("test shapes:", X_test.shape, y_test.shape)


# 已完成任务 1：连接分类网络的前向计算和损失。
# 【目标】先计算训练数据上的初始 logits 和 loss，暂不训练。
# 【输入/输出】X_train (12,2)、y_train (12,)；logits (12,3)、loss 标量。
# 【步骤 1】设置 np.random.seed(42)，再创建 dense1 = Dense(2, 3)、
# relu = ReLU()、dense2 = Dense(3, 3) 和 loss_fn = SoftmaxCrossEntropyLoss()。
# 【步骤 2】依次用 dense1、relu、dense2 做前向计算，变量依次命名为
# Z1、A1、logits；直接把 logits 和 y_train 传给 loss_fn.forward 得到 loss。
# 【步骤 3】打印 logits.shape、loss，并检查 loss 是否有限（np.isfinite）。
# 【验收】logits.shape 为 (12,3)，loss 是有限标量；初始 loss 不必达到
# 特定数值或准确率。此时没有更新参数，结果不代表训练已完成。
np.random.seed(42)
dense1 = Dense(2,3)
relu = ReLU()
dense2 = Dense(3,3)
loss_fn = SoftmaxCrossEntropyLoss()
Z1 = dense1.forward(X_train)
A1 = relu.forward(Z1)
logits = dense2.forward(A1)
loss = loss_fn.forward(logits,y_train)
print("logits.shape:",logits.shape)
print("loss:",loss)
print(np.isfinite(loss))
# 已完成任务 2：连接反向传播和一次参数更新。
# 【目标】使用任务 1 的前向缓存，完成反向传播和一次参数更新，
# 再观察更新前后的训练损失；暂不写训练循环。
# 【输入/输出】已有 loss、loss_fn、dense1、relu、dense2；
# d_logits (12,3)、dA1 (12,3)、dZ1 (12,3)、dX (12,2)；
# updated_loss 为更新后的训练损失标量。
# 【步骤 1】依次调用 loss_fn、dense2、relu、dense1 的 backward，
# 把返回值分别保存为 d_logits、dA1、dZ1、dX；前一结果传给下一层。
# 【步骤 2】所有反向完成后，创建 optimizer = Adam(learning_rate=0.01)，
# 分别对 dense1 和 dense2 调用 optimizer.step，更新两层参数。
# 【步骤 3】更新后，重新用 X_train 依次经过 dense1、relu、dense2，
# 得到更新后的 logits，再调用 loss_fn.forward 得到 updated_loss。
# 打印各梯度的 shape、更新前的 loss 和更新后的 updated_loss。
# 【验收】四个梯度形状符合上面的约定；两次损失均有限，观察它们的变化。
# 一次更新后损失是否下降是观察结果，不作为预先保证。
d_logits = loss_fn.backward()
dA1 = dense2.backward(d_logits)
dZ1 = relu.backward(dA1)
dX = dense1.backward(dZ1)
print("d_logits.shape:", d_logits.shape)
print("dA1.shape:", dA1.shape)
print("dZ1.shape:", dZ1.shape)
print("dX.shape:", dX.shape)
optimizer = Adam(learning_rate=0.01)
optimizer.step(dense1)
optimizer.step(dense2)
Z1 = dense1.forward(X_train)
A1 = relu.forward(Z1)
logits = dense2.forward(A1)
updated_loss = loss_fn.forward(logits,y_train)
print("logits.shape:",logits.shape)
print("updated_loss:",updated_loss)


# 已完成任务 3：组织训练循环并观察损失。
# 【目标】在任务 2 已更新一次的网络上，继续用完整训练集训练 200 次，
# 观察训练损失的变化；本任务不使用独立评估集。
# 【输入/输出】复用已有 dense1、relu、dense2、loss_fn、optimizer，
# X_train (12,2)、y_train (12,)；每次得到 logits (12,3) 和 loss 标量。
# 【步骤 1】设置 num_epochs = 200，使用 for 循环依次完成每个 epoch。
# 每个 epoch 使用全部 12 个训练样本，因此每个 epoch 更新一次参数。
# 【步骤 2】每轮依次前向、计算 loss，再反向传播，最后更新两个 Dense。
# 可参考上方任务 1 和任务 2 的计算顺序；不要在循环内重新创建网络或优化器。
# 【步骤 3】在第 1、50、100、150、200 轮打印 epoch 和本轮 loss。
# 这里打印的是本轮更新前的 loss；epoch 编号从 1 开始显示。
# 【验收】脚本运行无异常，打印的损失均为有限数；比较第 1 轮和第 200 轮，
# 观察整体趋势，不要求每一轮都下降。循环前已有 1 次更新，循环中再更新
# 200 次，因此每层总共更新 201 次；准确率在任务 4 再计算。

for i in range(200):
    Z1 = dense1.forward(X_train)
    A1 = relu.forward(Z1)
    logits = dense2.forward(A1)
    loss = loss_fn.forward(logits,y_train)
    d_logits = loss_fn.backward()
    dA1 = dense2.backward(d_logits)
    dZ1 = relu.backward(dA1)
    dX = dense1.backward(dZ1)
    optimizer.step(dense1)
    optimizer.step(dense2)
    epoch = i + 1
    if epoch == 1 or epoch % 50 == 0:
        print("epoch:", epoch, "loss:", loss)


# 已完成任务 4：评估训练集与独立评估集。
# 【目标】用训练结束后的网络计算两个数据集的 loss 和分类准确率，
# 将独立评估集准确率与固定预测类别 0 的简单基线比较。
# 【输入/输出】X_train (12,2)、y_train (12,)；X_test (6,2)、y_test (6,)。
# train_logits (12,3)、test_logits (6,3)；train_predictions (12,)、
# test_predictions (6,)；train_loss、test_loss、train_accuracy、
# test_accuracy、baseline_accuracy 均为标量。
# 【步骤 1】训练完成后，分别让 X_train、X_test 经过 dense1、relu、dense2，
# 得到 train_logits、test_logits；分别与对应标签传给 loss_fn.forward，
# 得到 train_loss、test_loss。评估只前向，不调用 backward 或 optimizer.step。
# 不要直接使用循环中的 logits 或 loss：它们来自最后一次更新之前。
# 【步骤 2】对各自 logits 使用 np.argmax(..., axis=1)，得到每行分数最大的
# 类别索引 train_predictions、test_predictions。
# 【步骤 3】分别比较预测与对应的整数标签，用 np.mean 计算正确比例，
# 得到 train_accuracy 和 test_accuracy。
# 【步骤 4】假设独立评估集全部预测为类别 0，比较 y_test 与 0，
# 用 np.mean 计算 baseline_accuracy；本评估集三类均衡，基线为 1/3。
# 打印两个 logits 和预测的 shape、预测值、标签、loss、准确率和基线。
# 【验收】训练/评估的输出及预测形状与上面一致，loss 有限，准确率在
# 0 到 1 之间；比较 test_accuracy 与 baseline_accuracy。若未超过基线，
# 保留真实结果并分析，不预设准确率数值。独立评估集只有 6 个样本，
# 结果只能说明这组数据上的表现。

# 最后一轮的 loss 是更新前计算的，因此更新完成后重新前向评估。
train_logits = dense2.forward(relu.forward(dense1.forward(X_train)))
train_loss = loss_fn.forward(train_logits, y_train)
test_logits = dense2.forward(relu.forward(dense1.forward(X_test)))
test_loss = loss_fn.forward(test_logits, y_test)

# 每行取最大分数所在的列，得到对应样本的预测类别索引。
train_predictions = np.argmax(train_logits, axis=1)
test_predictions = np.argmax(test_logits, axis=1)

# 比较得到布尔数组；对它求平均，就是预测正确的样本比例。
train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)
# 基线始终预测类别 0；用同一评估集计算正确比例。
baseline_accuracy = np.mean(y_test == 0)

print("train logits and prediction shapes:", train_logits.shape, train_predictions.shape)
print("train predictions:", train_predictions, "labels:", y_train)
print("train loss:", train_loss, "accuracy:", train_accuracy)
print("test logits and prediction shapes:", test_logits.shape, test_predictions.shape)
print("test predictions:", test_predictions, "labels:", y_test)
print("test loss:", test_loss, "accuracy:", test_accuracy)
print("class-0 baseline accuracy:", baseline_accuracy)

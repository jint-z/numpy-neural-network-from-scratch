# NumPy 神经网络学习记录

这是一个从基础计算逐步搭建神经网络的学习项目。项目使用 NumPy 手写前向传播、反向传播和参数更新，通过编号实验记录从回归到多类别分类的学习过程。

重点是理解计算过程和观察模型行为；代码是教学实验，不是面向生产环境的神经网络框架。

## 项目结构

```text
src/
  layers.py       全连接层、ReLU 和 Softmax
  losses.py       均方误差、交叉熵和稳定的 Softmax 交叉熵组合损失
  optimizers.py   SGD、Momentum 和 Adam
experiments/      按编号递进的独立实验
AGENTS.md         协作方式与学习进度记录
```

## 实验路线

| 编号 | 内容 |
| --- | --- |
| 00–06 | NumPy 基础、神经元、全连接层和网络前向 |
| 07–12 | 均方误差、全连接层与 ReLU 的反向传播 |
| 13–17 | 数值梯度与梯度检查 |
| 18–20 | 参数更新、优化器与回归训练循环 |
| 21–26 | Softmax、交叉熵、分类梯度及稳定组合损失 |
| 27–29 | 分类训练、损失与准确率 |
| 30–33 | 新点预测、类别变化、ReLU 输出和梯度贡献 |

部分实验使用固定的合成数据；不同实验的随机种子和训练设置可能不同。部分注释和 TODO 保留了学习时的思考过程，运行结果需要结合具体实验判断。

## 运行

示例环境为 Python 3.12 和 NumPy 2.3.5。在项目根目录创建并启用虚拟环境后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

在项目根目录以模块方式运行某个实验，例如：

```powershell
python -m experiments.17_two_layer_network_gradient_check
python -m experiments.27_classification_training
python -m experiments.33_relu_gradient_contributions
```

实验主要通过打印数值和形状进行观察，并非完整的自动化测试套件。分类示例的数据量很小，结果只能说明对应实验中的表现，不能据此推断在真实数据上的泛化能力。

## 当前学习方向

在已实现的训练闭环上，继续观察损失、准确率、预测和梯度之间的关系，并逐步学习如何评估模型。项目会保留实验作为学习记录，随着理解的加深继续更新。

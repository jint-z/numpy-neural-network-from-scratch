import numpy as np
from src.layers import Dense,ReLU,Softmax
from src.optimizers import Adam
from src.losses import MSELoss

np.random.seed(42)
dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 1)
X = np.array([
    [1.0, 1.0],
    [1.0, 3.0],
    [2.0, 5.0],
    [3.0, 2.0],
    [4.0, 1.0],
    [4.0, 6.0],
    [5.0, 3.0],
    [6.0, 2.0]
])

y_true = np.array([
    [2.0],
    [4.0],
    [7.0],
    [5.0],
    [5.0],
    [10.0],
    [8.0],
    [8.0]
])

loss_fn = MSELoss()
original_lr = 0.01
optimizer = Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)
descend_rate = 0.001


num_epochs = 500
batch_size = 4
np.random.seed(42)
dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 1)
num_samples = X.shape[0]

for epoch in range(num_epochs):

    # 1. 每个 epoch 开始时重新打乱数据
    indices = np.random.permutation(num_samples)

    X_shuffled = X[indices]
    y_shuffled = y_true[indices]

    # 2. 遍历每个 mini-batch
    for start in range(0, num_samples, batch_size):
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # 3. forward
        Z1 = dense1.forward(X_batch)
        A1 = relu.forward(Z1)
        y_pred = dense2.forward(A1)

        # 4. loss
        loss = loss_fn.forward(y_pred,y_batch)

        # 5. backward
        d_y_pred = loss_fn.backward(y_pred,y_batch)
        dA1 = dense2.backward(d_y_pred)
        dZ1 = relu.backward(dA1)
        dX = dense1.backward(dZ1)

        # 6. 参数更新
        optimizer.step(dense1)
        optimizer.step(dense2)


#结果计算
Z1 = dense1.forward(X)
A1 = relu.forward(Z1)
y_pred = dense2.forward(A1)
loss = loss_fn.forward(y_pred,y_true)
# print("y_pred:")
# print(y_pred)
print("loss:")
print(loss)
#测试
X_test = np.array([
    [1.0, 4.0],
    [4.0, 1.0],
    [2.0, 2.0],
    [3.0, 5.0],
    [5.0, 2.0]
])
y_test = np.array([
    [5.0],
    [5.0],
    [4.0],
    [8.0],
    [7.0]
])
Z1_test = dense1.forward(X_test)
A1_test = relu.forward(Z1_test)
y_pred_test = dense2.forward(A1_test)

print("y_pred_test:")
print(y_pred_test)


X = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0]
])

softmax = Softmax()
output = softmax.forward(X)

print(output)
print("shape:", output.shape)
print("row sums:", np.sum(output, axis=1))

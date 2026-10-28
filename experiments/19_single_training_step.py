import numpy as np
from src.layers import Dense,ReLU

dense1 = Dense(2, 3)
relu = ReLU()
dense2 = Dense(3, 1)
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

y_true = np.array([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])

Z1 = dense1.forward(X)
A1 = relu.forward(Z1)
y_pred = dense2.forward(A1)
d_y_pred = 2*(y_pred-y_true)/y_pred.size
loss = np.mean((y_pred-y_true)**2)
dA1 = dense2.backward(d_y_pred)
dZ1 = relu.backward(dA1)
dX = dense1.backward(dZ1)
learning_rate = 0.1
dense1.W = dense1.W - learning_rate * dense1.dW
dense1.b = dense1.b - learning_rate *dense1.db
dense2.W = dense2.W - learning_rate * dense2.dW
dense2.b = dense2.b - learning_rate *dense2.db

print("loss:",loss,"\n")
print("y_pred:",y_pred,"\n")
print("dense1.dW.shape:",dense1.dW.shape)
print("dense1.db.shape:",dense1.db.shape)
print("dense2.dW.shape:",dense2.dW.shape)
print("dense2.db.shape:",dense2.db.shape)

loss_before = loss
Z1 = dense1.forward(X)
A1 = relu.forward(Z1)
new_y_pred = dense2.forward(A1)
loss_after = np.mean((new_y_pred-y_true)**2)

print("loss_before:",loss_before)
print("loss_after:",loss_after)
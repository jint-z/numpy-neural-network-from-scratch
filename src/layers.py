import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)
    def forward(self,X):
        self.X = X
        return X @ self.W + self.b
    def backward(self,dY):
        self.dW = self.X.T @ dY
        self.db = np.sum(dY,axis=0)
        dX = dY @ self.W.T
        return dX

        
class ReLU:
    def forward(self,X):
        self.X = X
        return np.maximum(X,0)
    def backward(self,dY):
        dX = dY*(self.X > 0)
        return dX


class Softmax:
    def forward(self, X):

        max_values = np.max(X,axis=1,keepdims=True)

        shifted_X = X - max_values

        exp_values = np.exp(shifted_X)

        sum_exp = np.sum(exp_values,axis=1,keepdims=True)

        output = exp_values / sum_exp

        return output
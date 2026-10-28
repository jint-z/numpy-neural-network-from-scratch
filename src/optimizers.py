import numpy as np
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layer):
        # 更新 layer.W
        layer.W = layer.W - self.learning_rate * layer.dW 
        # 更新 layer.b
        layer.b = layer.b - self.learning_rate * layer.db 


class Momentum:
    def __init__(self,learning_rate,beta):
        self.learning_rate = learning_rate
        self.beta = beta
    def step(self,layer):
        if not hasattr(layer, "vW"):
            layer.vW = np.zeros_like(layer.W)
            layer.vb = np.zeros_like(layer.b)

        layer.vW = self.beta * layer.vW + (1-self.beta) * layer.dW
        layer.vb = self.beta * layer.vb + (1-self.beta) * layer.db

        layer.W -= self.learning_rate * layer.vW
        layer.b -= self.learning_rate * layer.vb


class Adam:
    def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon =1e-8 ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self,layer):
        if not hasattr(layer,"mW"):
            layer.mW = np.zeros_like(layer.W)
            layer.mb = np.zeros_like(layer.b)
            layer.vW = np.zeros_like(layer.W)
            layer.vb = np.zeros_like(layer.b)
            layer.t  = 0
        layer.t += 1
        
        layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
        layer.mb = self.beta1 * layer.mb + (1 - self.beta1)* layer.db

        layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * layer.dW**2
        layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * layer.db**2

        mW_hat = layer.mW / (1 - self.beta1 ** layer.t)
        mb_hat = layer.mb / (1 - self.beta1 ** layer.t)
        
        vW_hat = layer.vW / (1 - self.beta2 ** layer.t)
        vb_hat = layer.vb / (1 - self.beta2 ** layer.t)

        layer.W -= self.learning_rate * mW_hat/(np.sqrt(vW_hat)+self.epsilon)
        layer.b -= self.learning_rate * mb_hat/(np.sqrt(vb_hat)+self.epsilon)
                
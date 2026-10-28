import numpy as np


#任务一
a=np.array(5)
b=np.array([1,2,3])
c=np.array([[1,2,3],[4,5,6]])
d=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print("a:", a)
print("a shape:", a.shape)
print("a ndim:", a.ndim)
#我预测 a shape: ()  a ndim: 0
print("b:", b)
print("b shape:", b.shape)
print("b ndim:", b.ndim)
#我预测 b shape: (3,)  b ndim: 1
print("c:", c)
print("c shape:", c.shape)
print("c ndim:", c.ndim)
#我预测 c shape: (2, 3)  c ndim: 2
print("d:", d)
print("d shape:", d.shape)
print("d ndim:", d.ndim)
#我预测 d shape: (2, 2, 3)  d ndim: 3

#任务二
x1=np.array([1,2,3])
x2=np.array([[1,2,3]])
x3=np.array([[1],
             [2],
             [3]])
#预测 x1,x2,x3的shape分别为（3，），（1，3），（3，1）
print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)
print("x3 shape:", x3.shape)


#任务三
X = np.array([[8,8,95],[7,9,90],[10,7,88],[5,10,77]])
W = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
b = np.array([[1,2]])
Y=X@W+b
print("X shape:", X.shape)
print("W shape:", W.shape)
print("b shape:", b.shape)
print("Y:")
print(Y)
print("Y shape:", Y.shape)

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

b = np.array([10, 20])
print(A+b)

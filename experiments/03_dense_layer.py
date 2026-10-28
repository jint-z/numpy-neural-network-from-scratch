import numpy as np

x = np.array([1.0, 2.0, 3.0])

W = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])

b = np.array([[1,2,3]])

output = x@W+b

print("x.shape:", x.shape)
print("W.shape:", W.shape)
print("b.shape:", b.shape)
print("output:", output)
print("output.shape:", output.shape)
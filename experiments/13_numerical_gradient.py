def f(x):
    return x ** 2


x = 3.0
eps = 1e-5

# 1. analytical gradient
analytical_grad = 2*x

# 2. numerical gradient
numerical_grad = (f(x+eps)-f(x-eps))/(2*eps)

print("analytical:", analytical_grad)
print("numerical :", numerical_grad)
print("difference:", abs(analytical_grad - numerical_grad))
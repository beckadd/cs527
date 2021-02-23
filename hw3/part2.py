import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt


def f(x):
    return np.square(x) * np.cos(x)


g = grad(f)
print(g(0.6))

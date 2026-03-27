import matplotlib.pyplot as plt
import numpy as np


# ボコボコした関数
def f(x):
    return np.sin(x) + 0.1 * x


# 勾配
def df(x):
    return np.cos(x) + 0.1


# 勾配=0になる場所を探す
x = np.linspace(-10, 10, 1000)
y = f(x)

plt.plot(x, y)
plt.axhline(y=0, color="gray", linestyle="--")
plt.grid()
plt.show()

import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -5.0から5.0まで0.1刻み
x = np.arange(-5.0, 5.0, 0.1)
# 0以下は0、0.1以上は1になるから・・[00000111111]みたいになるか
# y = step_function(x)
y = sigmoid(x)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7  # log(0)を防ぐための微小な値
    return -np.sum(t * np.log(y + delta))


t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# クラス「2」の確率が最も高い場合
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print("y1の誤差:", cross_entropy_error(y1, t))


# クラス「7」の確率が最も高い場合
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print("y2の誤差:", cross_entropy_error(y2, t))
# y2の方が誤差が大きいことがわかる
